"""Execute the promoted H confirmation workflow's exact aggregate compatibility guard."""
import ast
import base64
import contextlib
import hashlib
import json
from pathlib import Path
import re
import tempfile
import textwrap
import unittest
import urllib.parse
import zlib

OLD = "9ea481a878f57cdd0671c8b4d7ad1ac2b5c9643f368295a3daa3485d5c8789d8"
NEW = "7ede9cd0a40a585d11fc94d08738d8a74abc2607b252c3ccc6e4b4614ef6eba4"
REVISION = "736097e5298fb71037a420ec8bd635c73a0154dd"
# Exact authenticated source-commit wrapper; compressed to keep this fixture self-contained.
OLD_SOURCE = zlib.decompress(base64.b64decode("""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"""))


def repository_root():
    here = Path(__file__).resolve()
    candidates = [*here.parents, *here.parent.parent.glob("DoVIS*")]
    for root in candidates:
        if (root / ".github/workflows/l1-H-cohort-validation.yml").is_file() and (root / "experiments/mutual-understanding/mutual_lab/cohort_context_validation.py").is_file():
            return root
    raise FileNotFoundError("Promoted H confirmation workflow and package are required")


ROOT = repository_root()
WORKFLOW = Path(__file__).with_name("l1-H-cohort-validation.yml")
if not WORKFLOW.is_file():
    WORKFLOW = ROOT / ".github/workflows/l1-H-cohort-validation.yml"
PACKAGE = ROOT / "experiments/mutual-understanding/mutual_lab"


def actual_guard():
    matched = re.search(r"(?ms)^          # BEGIN fixed H aggregation compatibility\n(.*?)^          # END fixed H aggregation compatibility", WORKFLOW.read_text())
    if matched is None:
        raise AssertionError("The promoted H workflow compatibility guard is missing")
    return textwrap.dedent(matched.group(1))


def nonaggregate_ast(source):
    tree = ast.parse(source)
    nodes = [node for node in tree.body if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "aggregate_cohort_validations")]
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)


class HConfirmationCompatibilityTests(unittest.TestCase):
    def execute(self, *, old=OLD, requested=NEW, changed_source=None, drift=None, old_api_source=OLD_SOURCE):
        source = (PACKAGE / "cohort_context_validation.py").read_bytes()
        if changed_source is not None:
            source = changed_source
        current_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in PACKAGE.glob("*.py")}
        current_hashes["cohort_context_validation.py"] = hashlib.sha256(source).hexdigest()
        recorded = dict(current_hashes)
        recorded["cohort_context_validation.py"] = old
        if drift is not None:
            current_hashes[drift] = "0" * 64
        calls = []
        def github(path):
            calls.append(path)
            self.assertTrue(path.startswith("/contents/experiments/mutual-understanding/mutual_lab/cohort_context_validation.py?"))
            self.assertEqual(urllib.parse.parse_qs(urllib.parse.urlparse(path).query).get("ref"), [REVISION])
            return {"type": "file", "encoding": "base64", "content": base64.b64encode(old_api_source).decode()}
        request = {"source_revision": REVISION, "confirmation_implementation_sha256": requested}
        values = {"context": {"source_sha256": recorded}, "current_hashes": current_hashes,
                  "request": request, "github": github, "Path": Path, "hashlib": hashlib,
                  "ast": ast, "base64": base64, "urllib": urllib, "json": json}
        error = None
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "mutual_lab"
            package.mkdir()
            (package / "cohort_context_validation.py").write_bytes(source)
            with contextlib.chdir(temporary):
                try:
                    exec(compile(actual_guard(), str(WORKFLOW) + ":compatibility", "exec"), values)
                except Exception as raised:
                    error = raised
        return values, error, calls

    def test_exact_known_pair_with_explicit_pin_and_immutable_source_ast_is_accepted(self):
        self.assertEqual(hashlib.sha256(OLD_SOURCE).hexdigest(), OLD)
        self.assertEqual(hashlib.sha256((PACKAGE / "cohort_context_validation.py").read_bytes()).hexdigest(), NEW)
        values, error, calls = self.execute()
        self.assertIsNone(error)
        self.assertEqual(len(calls), 1)
        self.assertEqual(values["compatibility"]["source_sha256"], OLD)
        self.assertEqual(values["compatibility"]["confirmation_sha256"], NEW)
        self.assertTrue(values["compatibility"]["native_training_functions_ast_identical"])
        self.assertEqual(values["compatibility"]["other_package_files_byte_identical"], 41)

    def test_all_training_and_other_top_level_ast_nodes_are_identical(self):
        source = (PACKAGE / "cohort_context_validation.py").read_bytes()
        # AST layouts vary by Python release; compare both authenticated
        # snapshots under the same interpreter instead of pinning a 3.12 dump.
        self.assertEqual(nonaggregate_ast(source), nonaggregate_ast(OLD_SOURCE))
        old_aggregate = [n for n in ast.parse(OLD_SOURCE).body if isinstance(n, ast.FunctionDef) and n.name == "aggregate_cohort_validations"]
        new_aggregate = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "aggregate_cohort_validations"]
        self.assertEqual(len(old_aggregate), 1)
        self.assertEqual(len(new_aggregate), 1)
        self.assertNotEqual(ast.dump(old_aggregate[0]), ast.dump(new_aggregate[0]))

    def test_unexpected_old_or_new_wrapper_or_request_pin_is_rejected(self):
        current = (PACKAGE / "cohort_context_validation.py").read_bytes()
        for arguments in ({"old": "0"*64}, {"requested": "0"*64},
                          {"changed_source": current + b"\n# unreviewed wrapper bytes\n"}):
            with self.subTest(arguments=list(arguments)):
                _, error, _ = self.execute(**arguments)
                self.assertIsInstance(error, ValueError)

    def test_unverifiable_immutable_old_source_bytes_are_rejected(self):
        _, error, _ = self.execute(old_api_source=OLD_SOURCE + b"\n# changed\n")
        self.assertIsInstance(error, ValueError)

    def test_every_other_frozen_package_file_remains_exact(self):
        names = sorted(p.name for p in PACKAGE.glob("*.py") if p.name != "cohort_context_validation.py")
        self.assertEqual(len(names), 41)
        for name in names:
            with self.subTest(name=name):
                _, error, _ = self.execute(drift=name)
                self.assertIsInstance(error, ValueError)


if __name__ == "__main__":
    unittest.main()
