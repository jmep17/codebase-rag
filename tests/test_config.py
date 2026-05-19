from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from codebase_rag import config, index


class ConfigPathTests(unittest.TestCase):
    def test_codebase_rag_home_controls_state_paths(self):
        with mock.patch.dict(os.environ, {"CODEBASE_RAG_HOME": "/tmp/cbr-state"}):
            self.assertEqual(config.data_home(), Path("/tmp/cbr-state"))
            self.assertEqual(config.default_db_path(), Path("/tmp/cbr-state/db"))
            self.assertEqual(config.meta_root(), Path("/tmp/cbr-state/meta"))
            self.assertEqual(config.default_user_skill_dir(), Path("/tmp/cbr-state/skills"))

    def test_project_meta_dir_uses_configured_state_root(self):
        with mock.patch.dict(os.environ, {"CODEBASE_RAG_HOME": "/tmp/cbr-state"}):
            meta_dir = index.project_meta_dir(Path("/tmp/example-project"))
        self.assertEqual(meta_dir.parent, Path("/tmp/cbr-state/meta"))
