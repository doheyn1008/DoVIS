"""Small-model invariants; never repeat the production resource benchmark."""

from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F
from native_lm import (BOS, EOS, PAD, ByteTokenizer, ModelConfig, NativeDocument,
    build_model, document_split, document_window, next_token_loss)


class CausalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.config = ModelConfig(id="small-invariant-fixture", hidden_dim=24, layers=2,
            heads=3, feedforward_dim=48, sequence_length=16)

    def test_initialization_is_repeatable_seed_sensitive_and_restores_caller_rng(self):
        before = torch.random.get_rng_state().clone()
        a = build_model(self.config, 31)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        torch.rand(19)
        b, c = build_model(self.config, 31), build_model(self.config, 32)
        self.assertTrue(all(torch.equal(value, b.state_dict()[name]) for name, value in a.state_dict().items()))
        self.assertFalse(torch.equal(a.embedding.weight, c.embedding.weight))
        self.assertIs(a.embedding.weight, a.output.weight)

    def test_future_tokens_do_not_change_prefix_logits_or_backpropagate_into_future_input_states(self):
        model = build_model(self.config, 33)
        a = torch.tensor([[BOS, 7, 8, 9, 10, EOS]])
        b = torch.tensor([[BOS, 7, 8, 77, 88, EOS]])
        self.assertTrue(torch.equal(model(a)[:, :3], model(b)[:, :3]))
        embeddings = []

        def retain(_module, _args, output):
            output.retain_grad(); embeddings.append(output)

        with model.embedding.register_forward_hook(retain):
            model(a)[:, :3, 17].sum().backward()
        self.assertEqual(float(embeddings[0].grad[:, 3:].abs().sum()), 0.)
        self.assertGreater(float(embeddings[0].grad[:, :3].abs().sum()), 0.)

    def test_right_padding_is_finite_masked_and_excluded_from_next_token_loss(self):
        model = build_model(self.config, 34)
        short = torch.tensor([[BOS, 7, 8, EOS]])
        padded = F.pad(short, (0, 4), value=PAD)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(short), model(padded)[:, :4], atol=1e-7, rtol=1e-6))
            self.assertTrue(torch.equal(model(padded)[:, 4:], torch.zeros(1, 4, 259)))
            self.assertTrue(torch.allclose(next_token_loss(model, short), next_token_loss(model, padded), atol=1e-7))
        loss = next_token_loss(model, padded)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        for ids in (torch.tensor([[PAD, PAD]]), torch.tensor([[BOS, PAD, 7]]),
                    torch.tensor([[BOS, 4, PAD, 9]]), torch.tensor([[BOS, 259]])):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                next_token_loss(model, ids)

    def test_real_adam_step_updates_a_small_scratch_model_on_a_declared_native_fixture(self):
        model = build_model(self.config, 35)
        document = CorpusBoundaryTests.document("a b.")
        tokens = document_window([document], "en", sequence_length=16)
        before = model.embedding.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        optimizer.zero_grad(set_to_none=True)
        next_token_loss(model, tokens).backward(); optimizer.step()
        self.assertFalse(torch.equal(before, model.embedding.weight))
        self.assertTrue(optimizer.state)


class CorpusBoundaryTests(unittest.TestCase):
    @staticmethod
    def document(text="A short native fixture.", language="en", **changes):
        record = NativeDocument(document_id="fixture", language=language, text=text,
            source_url="https://example.invalid/unit-test-only", revision="fixture-v1", license_id="CC0-1.0",
            attribution="Test-authored fixture, never a downloaded corpus", text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            near_duplicate_group="fixture-group", rights_review="approved", foreign_passage_audit="passed")
        return replace(record, **changes)

    def test_fixed_byte_map_roundtrips_every_native_script_without_fitting(self):
        tokenizer = ByteTokenizer()
        for text in ("한국어 문장.", "日本語の文。", "An English sentence.", "Una oración.", "😀"):
            encoded = tokenizer.encode(text)
            self.assertEqual(encoded[0], BOS)
            self.assertEqual(encoded[-1], EOS)
            self.assertEqual(len(encoded), len(text.encode("utf-8")) + 2)
            self.assertEqual(tokenizer.decode(encoded), text)
        self.assertEqual(tokenizer.vocab_size, 259)
        self.assertFalse(hasattr(tokenizer, "fit"))

    def test_foreign_or_unreviewed_records_fail_before_any_batch_text_is_encoded(self):
        good = self.document("English fixture.")
        invalid = (self.document("한국어", "ko"), self.document(foreign_passage_audit="pending"),
                   self.document(rights_review="pending"), self.document(license_id="unknown"),
                   self.document(text_sha256="changed"))
        for record in invalid:
            with self.subTest(record=record), patch.object(ByteTokenizer, "encode", wraps=ByteTokenizer.encode) as encode:
                with self.assertRaises(ValueError):
                    document_window([good, record], "en")
                encode.assert_not_called()

    def test_duplicate_group_split_is_stable_and_document_windows_never_pack_foreign_or_other_documents(self):
        first, second = self.document("abc"), self.document("xyz", document_id="second")
        self.assertEqual(document_split(first), document_split(second))
        self.assertEqual(document_split(first), document_split(first))
        windows = document_window([first, second], "en", sequence_length=8)
        self.assertEqual(ByteTokenizer.decode(windows[0].tolist()), "abc")
        self.assertEqual(ByteTokenizer.decode(windows[1].tolist()), "xyz")
        self.assertTrue(bool((windows[:, 5:] == PAD).all()))
        with self.assertRaisesRegex(ValueError, "no next-token target"):
            document_window([first], "en", sequence_length=8, window=1)


if __name__ == "__main__":
    unittest.main()
