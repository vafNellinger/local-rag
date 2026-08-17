"""Tests für plattformkorrekte Speicherorte und die Index-Migration.

Der Zweck: Index und Dokumente liegen in *data*, Einstellungen in *config*,
Wegwerfbares in *cache* — je Plattform am richtigen Ort (platformdirs). Und ein
Altbestand vom früheren Sammelort wandert einmalig, ohne etwas zu überschreiben.
"""

from __future__ import annotations

from pathlib import Path

import platformdirs

from rag import paths


class TestPfade:
    def test_dirs_liegen_unter_platformdirs(self):
        assert paths.DATA_DIR == Path(platformdirs.user_data_dir("local-rag"))
        assert paths.CACHE_DIR == Path(platformdirs.user_cache_dir("local-rag"))
        assert paths.CONFIG_DIR == Path(platformdirs.user_config_dir("local-rag"))

    def test_index_und_dokumente_sind_data(self):
        # Der Index darf nicht in den Cache — ein Cache-Cleaner würde ihn löschen.
        assert paths.DEFAULT_INDEX_PATH.parent == paths.DATA_DIR
        assert paths.UPLOAD_DIR.parent == paths.DATA_DIR

    def test_settings_sind_config(self):
        assert paths.SETTINGS_PATH.parent == paths.CONFIG_DIR

    def test_wegwerfbares_ist_cache(self):
        for p in (
            paths.EXTRACT_CACHE_DIR,
            paths.ONNX_DIR,
            paths.HF_LANGUAGE_CACHE,
            paths.LOG_PATH,
        ):
            assert p.parent == paths.CACHE_DIR


class TestMigration:
    def _alt_index(self, root: Path) -> Path:
        alt = root / "cache" / "index.db"
        alt.parent.mkdir(parents=True)
        alt.write_text("db")
        (alt.parent / "index.db-wal").write_text("wal")
        (alt.parent / "index.db-shm").write_text("shm")
        return alt

    def test_verschiebt_index_samt_wal_und_shm(self, tmp_path):
        alt = self._alt_index(tmp_path)
        ziel = tmp_path / "data" / "index.db"
        paths._migrate_legacy(alt=alt, ziel=ziel)
        assert ziel.read_text() == "db"
        assert (ziel.parent / "index.db-wal").read_text() == "wal"
        assert (ziel.parent / "index.db-shm").read_text() == "shm"
        assert not alt.exists()

    def test_idempotent_wenn_ziel_schon_existiert(self, tmp_path):
        alt = self._alt_index(tmp_path)
        ziel = tmp_path / "data" / "index.db"
        ziel.parent.mkdir(parents=True)
        ziel.write_text("bestehender index")
        paths._migrate_legacy(alt=alt, ziel=ziel)
        # Nichts überschreiben: Ziel bleibt, Altbestand bleibt liegen.
        assert ziel.read_text() == "bestehender index"
        assert alt.exists()

    def test_ohne_altbestand_kein_fehler(self, tmp_path):
        alt = tmp_path / "cache" / "index.db"  # existiert nicht
        ziel = tmp_path / "data" / "index.db"
        paths._migrate_legacy(alt=alt, ziel=ziel)
        assert not ziel.exists()
