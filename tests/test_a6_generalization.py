"""CPU checks for the new algorithms and release/checkpoint interface."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from experiments.a6_generalization.common import Config
from experiments.a6_generalization import models,optimizers,training
from experiments.a6_generalization.analysis import ensemble_losses

class GeneralizationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_architecture_carryovers_and_attention_only_dropout(self):
        for kind in ('dense','moe','loop'):
            c=Config(architecture=kind,width=64,depth=8,context=16,attention_dropout=.15)
            m=models.build(c);ids=torch.randint(0,19,(2,16));loss=training.nll(m(ids),ids);loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(b.self_attn.dropout==.15 and b.dropout==0 for b in m.model.layers))
            if hasattr(m,'finish_step'):m.finish_step()

    def test_fine_tuning_freezes_and_transfer(self):
        c=Config(width=64,depth=2,context=16)
        original=models.build(c)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'checkpoints').mkdir()
            torch.save({'model':original.state_dict()},root/'checkpoints/related.pt')
            for mode in ('full','toward_initialization','frozen_head','frozen_body'):
                m=models.build(replace(c,initialization='related',adaptation=mode),root)
                names={n for n,p in m.named_parameters() if p.requires_grad}
                if mode=='frozen_head':self.assertNotIn('lm_head.weight',names);self.assertIn('model.embed_tokens.weight',names)
                elif mode=='frozen_body':self.assertEqual(names,{'model.embed_tokens.weight','lm_head.weight'})
                else:self.assertEqual(len(names),len(list(m.parameters())))
                for n,p in m.named_parameters():torch.testing.assert_close(p,original.state_dict()[n])

    def test_hyperball_preserves_linear_norms_and_updates_embedding(self):
        m=models.build(Config(width=64,depth=2,context=16))
        weights=[x.weight for x in m.modules() if isinstance(x,torch.nn.Linear)]
        radii=[p.norm().item() for p in weights];embedding=m.model.embed_tokens.weight.detach().clone()
        ids=torch.randint(0,19,(2,16));training.nll(m(ids),ids).backward()
        optimizers.Hyperball(m,.03).step()
        np.testing.assert_allclose([p.norm().item() for p in weights],radii,rtol=2e-6)
        self.assertFalse(torch.equal(embedding,m.model.embed_tokens.weight))

    def test_sam_zero_radius_matches_adamw_with_microbatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'splits').mkdir()
            rng=np.random.default_rng(4)
            np.save(root/'splits/target_30k.npy',rng.integers(0,19,(16,16),dtype=np.uint8))
            np.save(root/'splits/validation.npy',rng.integers(0,19,(4,16),dtype=np.uint8))
            base=Config(width=64,depth=2,context=16,ntrain=16,batch=8,micro_batch=4,epochs=1,attention_dropout=.15)
            for mode in ('adamw','sam'):
                training.train(replace(base,optimizer=mode,rho=0),root=root,output=root/mode,device='cpu')
            states=[torch.load(root/m/'model.pt',weights_only=False)['model'] for m in ('adamw','sam')]
            for n in states[0]:torch.testing.assert_close(states[0][n],states[1][n],rtol=1e-6,atol=1e-7)

    def test_sharpness_probe_restores_weights_and_mode(self):
        from experiments.a6_generalization.p3_optimization import directional_sharpness
        m=models.build(Config(width=64,depth=2,context=16)).train()
        saved={n:p.detach().clone() for n,p in m.named_parameters()}
        result=directional_sharpness(m,torch.randint(0,19,(2,16)))
        self.assertTrue(np.isfinite(result['max_loss_increase']))
        self.assertTrue(m.training)
        for n,p in m.named_parameters():torch.testing.assert_close(p,saved[n],rtol=0,atol=0)

    def test_probability_ensembles_singleton_and_identical_members(self):
        z=np.random.default_rng(42).normal(size=(2,4,19));labels=np.array([[1,2,3,4],[4,3,2,1]])
        losses=ensemble_losses([z,z,z],labels)
        np.testing.assert_allclose(losses[0],losses[-1],rtol=1e-12)
        self.assertEqual(losses.shape,(3,2))

    def test_ensemble_averages_probabilities_not_logits(self):
        logits=[np.array([[[8.,0.]]]),np.array([[[0.,2.]]])]
        actual=ensemble_losses(logits,np.array([[0]]))[1,0]
        probability=(1/(1+np.exp(-8))+1/(1+np.exp(2)))/2
        self.assertAlmostEqual(actual,-np.log(probability),places=12)
        self.assertNotAlmostEqual(actual,np.log1p(np.exp(-3)),places=3)

if __name__=='__main__':unittest.main()
