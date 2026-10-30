"""Dataset-free correctness checks for the released mechanisms."""
import math
import hashlib
import gzip
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
import torch
from torch.nn import functional as F
from modeling import AutoregressiveLM,initialize_model
from model_config import LMConfig
from experiments.a5_capacity.p1_depth_width import grid,parameters,training_flops
from experiments.a5_capacity.p2_readout import HeadLM
from experiments.a5_capacity.p2_readout import CPLM
from experiments.a5_capacity.p3_moe import RouterMLP
from experiments.a5_capacity.p4_looping import LoopLM
from experiments.a5_capacity.diagnostics import router_metrics,gradient_energy,signal_profile

torch.set_num_threads(2)


def tiny(depth=2):
    return LMConfig('tiny',32,8,16,32,depth,2,2,head_dim=8)


class Algorithms(unittest.TestCase):
    def test_raw_reference_checkpoint(self):
        from experiments.a5_capacity.p1_depth_width import load_model
        cfg=tiny();model=AutoregressiveLM(cfg).eval();initialize_model(model)
        x=torch.randint(cfg.vocab_size,(1,8))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt';torch.save(model.state_dict(),path)
            row=dict(architecture=cfg.to_dict(),checkpoint=path.name,
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            restored=load_model(row,root=folder,device='cpu')
            torch.testing.assert_close(model(x),restored(x),rtol=0,atol=0)
            del restored  # Release mmap storage before deleting files on NFS.

    def test_local_reference_manifest_and_history(self):
        from experiments.a5_capacity.p1_depth_width import best, runs, load_history, manifest
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaisesRegex(FileNotFoundError,'publication is TODO'):
                manifest(root)
            rows=[dict(depth=8,width=512,seed=42,val_loss=loss,checkpoint=checkpoint,
                       cell_id=str(i),history='history.json.gz')
                  for i,(loss,checkpoint) in enumerate([(3.,None),(4.,'model.pt'),(float('inf'),'bad.pt')])]
            (root/'manifest.json').write_text(json.dumps(dict(runs=rows)))
            with gzip.open(root/'history.json.gz','wt') as f:json.dump([dict(step=1,val_loss=4.)],f)
            self.assertEqual(len(runs(root=root)),3)
            self.assertEqual(best(root=root)['cell_id'],'1')
            self.assertEqual(load_history(root=root),[dict(step=1,val_loss=4.)])

    def test_native_checkpoint_builder_paths(self):
        from experiments.a5_capacity.sweeps import VARIANTS, config
        from train import build_model
        from model_io import save_model, load_model
        x=torch.randint(32,(1,8))
        with tempfile.TemporaryDirectory() as folder:
            for name in VARIANTS:
                cfg=config(name,model_config=tiny(8),precision='fp32')
                model=build_model(cfg,device='cpu').eval()
                save_model(model,folder,metadata=dict(model_builder=cfg.model_builder,
                           model_builder_kwargs=cfg.model_builder_kwargs))
                restored=load_model(folder,device='cpu',precision='fp32')
                # Random routing draws fresh assignments on each forward.
                torch.manual_seed(7);expected=model(x)
                torch.manual_seed(7);actual=restored(x)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                del restored

    def test_shape_accounting(self):
        for cfg in grid():
            with torch.device('meta'):model=AutoregressiveLM(cfg)
            self.assertEqual(sum(p.numel() for p in model.parameters()),parameters(cfg))

    def test_router_metrics(self):
        s=router_metrics(torch.full((4,4),.25),torch.arange(4)[:,None])
        self.assertEqual(s['load_cv'],0);self.assertAlmostEqual(s['entropy'],1)
        self.assertEqual(s['mean_max_probability'],.25)
        s=router_metrics(torch.tensor([[1.,0,0,0]]).repeat(4,1),torch.zeros(4,1,dtype=torch.long))
        self.assertAlmostEqual(s['load_cv'],math.sqrt(3),places=6)
        self.assertEqual(s['entropy'],0);self.assertEqual(s['mean_max_probability'],1)

    def test_packed_experts_outputs_and_gradients(self):
        for skew in (False,True):
            for k in (1,2,4):
                m=RouterMLP(tiny(),top_k=k,balance=0.).eval()
                x=torch.randn(2,8,16,requires_grad=True)
                if skew:
                    with torch.no_grad():x[:,:,0]=20;m.router.weight[:,0]=torch.arange(4.)
                reference=deepcopy(m)
                z=x.detach().clone().requires_grad_()
                p=reference.router(z.flatten(0,1)).float().softmax(-1)
                indices=p.topk(k,-1).indices
                vals=[]
                for t,row in enumerate(z.flatten(0,1)):
                    vals.append(sum((4/k)*p[t,e]*((F.silu(row@reference.experts.gate[e])
                        *(row@reference.experts.up[e]))@reference.experts.down[e]) for e in indices[t]))
                expected=torch.stack(vals).reshape_as(z)
                actual=m(x)
                torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)
                actual.square().sum().backward();expected.square().sum().backward()
                torch.testing.assert_close(x.grad,z.grad,rtol=2e-4,atol=2e-4)
                for a,b in zip(m.parameters(),reference.parameters()):
                    if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=2e-4,atol=2e-4)

    def test_quantile_accumulation_and_eval(self):
        a=RouterMLP(tiny(),quantile=True,balance=0.)
        b=deepcopy(a);x=torch.randn(4,8,16)
        a(x);b(x[:1]);b(x[1:])
        a.finish_step();b.finish_step()
        torch.testing.assert_close(a.bias,b.bias,rtol=0,atol=0)
        a.eval();bias=a.bias.clone()
        y=a(x);changed=x.clone();changed[:,-2:]*=10
        torch.testing.assert_close(y[:,:-2],a(changed)[:,:-2])
        a.finish_step();torch.testing.assert_close(a.bias,bias)

    def test_unit_weights_and_frozen_norm(self):
        m=RouterMLP(tiny(),unit=True,balance=0.)
        m(torch.randn(2,8,16)).sum().backward()
        self.assertIsNone(m.router.weight.grad)
        m=RouterMLP(tiny(),routing='frozen',norm=True)
        self.assertFalse(m.router.weight.requires_grad)
        self.assertTrue(m.norm.weight.requires_grad)

    def test_heads_and_cplm(self):
        x=torch.randint(32,(2,8))
        for method in ('softmax','sigsoftmax','mos'):
            model=HeadLM(tiny(),rank=4,method=method).eval()
            initialize_model(model);model.post_initialize()
            y=model(x)
            self.assertTrue(torch.isfinite(y).all())
            y.square().mean().backward()
        for rank in (None,4):
            model=CPLM(tiny(),rank=rank).eval()
            initialize_model(model);model.post_initialize()
            y=model(x)
            torch.testing.assert_close(y.exp().sum(-1),torch.ones(2,8),rtol=1e-5,atol=1e-5)
            changed=x.clone();changed[:,-2:]=(changed[:,-2:]+1)%32
            torch.testing.assert_close(y[:,:-2],model(changed)[:,:-2],rtol=0,atol=0)
            F.cross_entropy(y[:,:-1].flatten(0,1),x[:,1:].flatten()).backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_loop_gradients_and_causality(self):
        x=torch.randint(32,(2,8))
        for passes in (1,2,3,6):
            model=LoopLM(tiny(8),passes=passes,boundary_norm=True,recall=.1)
            initialize_model(model);model.eval()
            y=model(x);changed=x.clone();changed[:,-2:]=(changed[:,-2:]+1)%32
            torch.testing.assert_close(y[:,:-2],model(changed)[:,:-2],rtol=0,atol=0)
            rows=signal_profile(model,x)
            self.assertEqual(len(rows),passes+2)
            self.assertTrue(all(s['gradient_rms']>0 for s in rows))

    def test_one_pass_matches_native_d8(self):
        cfg=tiny(8);x=torch.randint(32,(2,8))
        torch.manual_seed(42);native=AutoregressiveLM(cfg);initialize_model(native)
        torch.manual_seed(42);loop=LoopLM(cfg,passes=1);initialize_model(loop)
        torch.testing.assert_close(native(x),loop(x),rtol=0,atol=0)

    def test_one_block_ladder_and_noninvasive_signals(self):
        cfg=tiny(8);x=torch.randint(32,(2,8));initial=None;count=None
        for passes in (1,2,3,4,6,18,20):
            torch.manual_seed(42)
            model=LoopLM(cfg,passes=passes,core_size=1,boundary_norm=True,recall=1.)
            initialize_model(model)
            core=model.model.layers[1]
            for block in model.model.layers[1:-1]:self.assertIs(block,core)
            self.assertEqual(sum(p.numel() for p in model.boundary.parameters()),0)
            n=sum(p.numel() for p in model.parameters())
            if initial is not None:
                self.assertEqual(n,count)
                for key,value in core.state_dict().items():
                    torch.testing.assert_close(value,initial[key],rtol=0,atol=0)
            initial=deepcopy(core.state_dict());count=n
            model(x).square().mean().backward()
            before=[p.grad.clone() for p in model.parameters()]
            with torch.no_grad():rows=signal_profile(model,x)
            self.assertEqual(len(rows),passes+2)
            self.assertTrue(model.training)
            self.assertTrue(all(math.isfinite(s['gradient_rms']) for s in rows))
            for p,g in zip(model.parameters(),before):
                torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
            model.eval();model.eval_passes=3
            self.assertEqual(len(signal_profile(model,x)),5)
            self.assertEqual(model.executed_depth,5)
            self.assertEqual(sum(p.numel() for p in model.parameters()),count)
            self.assertGreater(training_flops(cfg,depth=22),training_flops(cfg))

    def test_boundary_formula_includes_final_recall(self):
        model=LoopLM(tiny(8),passes=2,core_size=1,boundary_norm=True,recall=1.).eval()
        initialize_model(model);x=torch.randint(32,(2,8))
        h=model.model.embed_tokens(x);rope=model.model.rotary_emb(h,torch.arange(8)[None])
        anchor=model.model.layers[0](h,rope);h=anchor
        for _ in range(2):
            h=model.model.layers[1](h,rope)
            h=h*(h.float().square().mean(-1,keepdim=True)+1e-5).rsqrt()+anchor
        expected=model.lm_head(model.model.norm(model.model.layers[-1](h,rope)))
        torch.testing.assert_close(model(x),expected,rtol=2e-5,atol=2e-5)

    def test_current_presets_are_isolated(self):
        from experiments.a5_capacity.sweeps import config
        for name,options in [('moe-norm',dict(norm=True)),('moe-noisy',dict(noise=.05)),
                             ('moe-cloned',dict(clone=True)),('moe-quantile',dict(quantile=True))]:
            self.assertEqual(config(name).model_builder_kwargs['router'],dict(options,balance=0.))
        for rank in (32,64,128,256):
            for method in ('softmax','mos','sigsoftmax','cplm'):
                self.assertEqual(config(f'{method}-r{rank}').model_builder_kwargs['rank'],rank)
        for name in ('mos','sigsoftmax','cplm'):
            self.assertNotIn('rank',config(name).model_builder_kwargs)
        self.assertEqual(config('loop-1block-r18').model_builder_kwargs,dict(passes=18,core_size=1))

    def test_projection_energy(self):
        weight=torch.eye(8)[:,:3]
        out=gradient_energy(torch.zeros(2,8),torch.tensor([0,1]),weight)
        g=torch.full((2,8),.125);g[0,0]-=1;g[1,1]-=1
        expected=g[:,3:].square().sum()/g.square().sum()
        self.assertAlmostEqual(out['suppressed_fraction'],float(expected),places=6)
        logits=torch.randn(5,8);weight=torch.randn(8,3);targets=torch.arange(5)
        plain=gradient_energy(logits,targets,weight)
        with torch.autocast('cpu',dtype=torch.bfloat16):mixed=gradient_energy(logits,targets,weight)
        self.assertEqual(plain,mixed)
        saturated=torch.tensor([[1000.]+[-1000.]*7])
        self.assertIsNone(gradient_energy(saturated,torch.tensor([0]),weight)['suppressed_fraction'])


if __name__=='__main__':unittest.main()
