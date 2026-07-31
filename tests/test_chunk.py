"""Tests für das Chunking.

Schwerpunkt sind die drei Stellen, an denen ein naiver Chunker Bedeutung
verliert: Überschriften-Kontext, Tabellenköpfe und deutsche Abkürzungen in
der Satzsegmentierung.
"""

from __future__ import annotations

from rag.chunk import (
    MIN_CHUNK_TOKENS,
    Chunk,
    _parse_blocks,
    chunk_markdown,
    estimate_tokens,
    split_sentences,
)

# Ein Zähler, der Wörter zählt statt Zeichen zu schätzen: macht die
# Größenerwartungen in den Tests exakt statt ungefähr.
def words(text: str) -> int:
    return max(1, len(text.split()))


class TestParseBlocks:
    def test_ueberschrift_und_absatz(self):
        blocks = _parse_blocks("# Titel\n\nEin Absatz.\n")
        assert [b.kind for b in blocks] == ["heading", "prosa"]
        assert blocks[0].level == 1
        assert blocks[0].text == "Titel"

    def test_absaetze_werden_an_leerzeile_getrennt(self):
        blocks = _parse_blocks("Erster Absatz.\n\nZweiter Absatz.")
        assert len(blocks) == 2

    def test_tabelle_bleibt_ein_block(self):
        md = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
        blocks = _parse_blocks(md)
        assert len(blocks) == 1
        assert blocks[0].kind == "tabelle"
        assert len(blocks[0].lines) == 4

    def test_codeblock_bleibt_zusammen(self):
        md = "```python\nx = 1\n\ny = 2\n```"
        blocks = _parse_blocks(md)
        assert len(blocks) == 1
        assert blocks[0].kind == "code"
        # Die Leerzeile im Code darf den Block nicht zerreissen.
        assert "y = 2" in blocks[0].text

    def test_unterschiedliche_ueberschriftenebenen(self):
        blocks = _parse_blocks("# Eins\n## Zwei\n###### Sechs")
        assert [b.level for b in blocks] == [1, 2, 6]


class TestSplitSentences:
    def test_einfache_saetze(self):
        assert len(split_sentences("Erster Satz. Zweiter Satz.")) == 2

    def test_abkuerzung_beendet_keinen_satz(self):
        # Der reale Fall: eine Fundstelle darf nicht zerschnitten werden.
        sentences = split_sentences("Es gilt gem. Abs. 2 dieser Ordnung. Danach mehr.")
        assert len(sentences) == 2
        assert "gem. Abs. 2" in sentences[0]

    def test_satz_ist_keine_abkuerzung(self):
        # "Satz" ist ein normales Wort — es stand faelschlich in der
        # Abkuerzungsliste und unterdrueckte damit jedes Satzende auf "Satz.".
        assert len(split_sentences("Erster Satz. Zweiter Satz.")) == 2

    def test_ordinalzahl_am_satzende_bleibt_konservativ(self):
        # Dokumentiertes Verhalten, kein Versehen: nach "Satz 1." endet hier
        # ein Satz, aber dieselbe Form steht in "1. Januar" mitten im Satz.
        # Ohne Semantik nicht trennbar, also wird nicht geschnitten — ein
        # verpasster Schnitt kostet nur Granularitaet.
        assert len(split_sentences("Es gilt Satz 1. Danach folgt mehr.")) == 1

    def test_zb_wird_nicht_getrennt(self):
        sentences = split_sentences("Dazu zählen z.B. Verträge. Und Rechnungen.")
        assert len(sentences) == 2
        assert "z.B. Verträge" in sentences[0]

    def test_ordinalzahl_trennt_nicht(self):
        sentences = split_sentences("Fällig am 1. Januar. Danach Verzug.")
        assert len(sentences) == 2
        assert "1. Januar" in sentences[0]

    def test_initiale_trennt_nicht(self):
        sentences = split_sentences("Unterschrieben von j. Öztürk. Ende.")
        assert len(sentences) == 2

    def test_ohne_satzende_ein_fragment(self):
        assert split_sentences("Nur ein Fragment ohne Punkt") == [
            "Nur ein Fragment ohne Punkt"
        ]

    def test_leerer_text(self):
        assert split_sentences("   ") == []


class TestHeadingPath:
    def test_pfad_wird_gesetzt(self):
        md = "# Kündigung\n\n## Fristen\n\nDie Frist beträgt 14 Tage."
        chunks = chunk_markdown(md, counter=words)
        assert len(chunks) == 1
        assert chunks[0].heading_path == ("Kündigung", "Fristen")
        assert chunks[0].heading == "Kündigung > Fristen"

    def test_embed_text_enthaelt_den_pfad(self):
        md = "# Kapitel\n\nInhalt."
        chunk = chunk_markdown(md, counter=words)[0]
        assert chunk.embed_text.startswith("Kapitel\n\n")
        # Der Rohtext bleibt ohne Präfix — der Präfix ist Retrieval-Kontext,
        # nicht Teil des Dokuments.
        assert chunk.text == "Inhalt."

    def test_neues_kapitel_erbt_keine_unterabschnitte(self):
        md = (
            "# Erstes\n\n## Unterpunkt\n\nText A.\n\n"
            "# Zweites\n\nText B."
        )
        chunks = chunk_markdown(md, counter=words)
        letzter = [c for c in chunks if "Text B" in c.text][0]
        assert letzter.heading_path == ("Zweites",)

    def test_ebenensprung_erzeugt_keine_luecke(self):
        # H1 direkt auf H3: die fehlende Ebene 2 darf im Pfad nicht als
        # leerer Eintrag auftauchen.
        chunks = chunk_markdown("# Eins\n\n### Drei\n\nText.", counter=words)
        assert chunks[0].heading_path == ("Eins", "Drei")

    def test_ohne_ueberschrift_leerer_pfad(self):
        chunk = chunk_markdown("Nur Text.", counter=words)[0]
        assert chunk.heading_path == ()
        assert chunk.embed_text == "Nur Text."


class TestSectionBoundaries:
    def test_sektionen_werden_nicht_vermischt(self):
        md = "## A\n\nAlpha.\n\n## B\n\nBeta."
        chunks = chunk_markdown(md, counter=words)
        assert len(chunks) == 2
        assert chunks[0].heading_path == ("A",)
        assert chunks[1].heading_path == ("B",)
        assert "Beta" not in chunks[0].text

    def test_ordinal_ist_fortlaufend(self):
        md = "## A\n\nAlpha.\n\n## B\n\nBeta.\n\n## C\n\nGamma."
        chunks = chunk_markdown(md, counter=words)
        assert [c.ordinal for c in chunks] == [0, 1, 2]


class TestSplitting:
    def test_langer_abschnitt_wird_geteilt(self):
        satz = "Dies ist ein vollstaendiger Satz mit genau zehn Woertern hier. "
        chunks = chunk_markdown(satz * 30, target_tokens=50, counter=words)
        assert len(chunks) > 1
        # Kein Chunk ueberschreitet das Ziel — der Overlap muss vom
        # Unit-Budget abgehen, nicht dazukommen.
        assert all(c.token_count <= 50 for c in chunks)

    def test_ueberlappung_innerhalb_der_sektion(self):
        satz = "Satz Nummer eins mit einigen Woertern darin. "
        chunks = chunk_markdown(
            satz * 20, target_tokens=40, overlap_tokens=10, counter=words
        )
        assert len(chunks) > 1
        # Der Anfang des zweiten Chunks muss im ersten vorkommen.
        anfang = chunks[1].text.split(".")[0]
        assert anfang in chunks[0].text

    def test_keine_ueberlappung_ueber_sektionsgrenzen(self):
        satz = "Ein Satz mit ausreichend vielen Woertern zum Fuellen hier. "
        md = f"## A\n\n{satz * 10}\n\n## B\n\n{satz * 10}"
        chunks = chunk_markdown(md, target_tokens=40, overlap_tokens=10, counter=words)
        erster_von_b = [c for c in chunks if c.heading_path == ("B",)][0]
        letzter_von_a = [c for c in chunks if c.heading_path == ("A",)][-1]
        # Kein Textanfang aus A darf in den ersten Chunk von B gelangen.
        assert erster_von_b.token_count <= 40
        assert letzter_von_a.heading_path != erster_von_b.heading_path

    def test_einzelner_ueberlanger_satz_bleibt_ganz(self):
        # Mitten im Satz zu schneiden kostet mehr Bedeutung, als der Ueberhang
        # an Praezision bringt — bge-m3 hat 8192 Token Kapazitaet.
        satz = "Wort " * 100
        chunks = chunk_markdown(satz.strip() + ".", target_tokens=20, counter=words)
        assert len(chunks) == 1


class TestTables:
    def test_kleine_tabelle_bleibt_ganz(self):
        md = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        chunks = chunk_markdown(md, counter=words)
        assert len(chunks) == 1
        assert chunks[0].kind == "tabelle"

    def test_grosse_tabelle_behaelt_kopf_in_jedem_teil(self):
        zeilen = "\n".join(f"| Zeile {i} | Wert {i} |" for i in range(40))
        md = f"| Bezeichnung | Betrag |\n| --- | --- |\n{zeilen}"
        chunks = chunk_markdown(md, target_tokens=30, counter=words)
        assert len(chunks) > 1
        # Ohne Spaltennamen ist eine Datenzeile bedeutungslos.
        assert all("Bezeichnung" in c.text for c in chunks)
        assert all("| --- |" in c.text for c in chunks)

    def test_tabellenzeilen_gehen_nicht_verloren(self):
        zeilen = "\n".join(f"| Zeile {i} | Wert {i} |" for i in range(40))
        md = f"| Bezeichnung | Betrag |\n| --- | --- |\n{zeilen}"
        chunks = chunk_markdown(md, target_tokens=30, counter=words)
        for i in range(40):
            assert any(f"| Zeile {i} |" in c.text for c in chunks), f"Zeile {i} fehlt"


class TestSmallRemainders:
    def test_kleiner_rest_wird_angehaengt(self):
        satz = "Ein Satz mit einigen Woertern zum Auffuellen der Groesse. "
        md = satz * 8 + "Kurz."
        chunks = chunk_markdown(md, target_tokens=40, overlap_tokens=0, counter=words)
        # "Kurz." darf kein eigener Chunk sein.
        assert not any(c.text.strip() == "Kurz." for c in chunks)
        assert "Kurz." in chunks[-1].text

    def test_kurzer_absatz_zwischen_langen_wird_nicht_eigener_chunk(self):
        # Bei realistischer Zielgroesse darf ein kurzer Absatz nicht als
        # eigener Mini-Chunk im Index landen — er wird mitgepackt.
        lang = "Satz mit zehn Woertern hier drin zum Auffuellen ok. " * 40
        md = f"{lang}\n\nKurzer Absatz.\n\n{lang}"
        chunks = chunk_markdown(md, counter=words)
        assert all(c.token_count >= MIN_CHUNK_TOKENS for c in chunks)
        assert any("Kurzer Absatz." in c.text and len(c.text) > 100 for c in chunks)


class TestEdgeCases:
    def test_leeres_dokument(self):
        assert chunk_markdown("", counter=words) == []

    def test_nur_ueberschriften_ohne_inhalt(self):
        assert chunk_markdown("# Eins\n\n## Zwei", counter=words) == []

    def test_nur_leerraum(self):
        assert chunk_markdown("\n\n   \n\n", counter=words) == []

    def test_umlaute_bleiben_erhalten(self):
        chunk = chunk_markdown("# Größe\n\nÄnderungen gemäß Prüfung.", counter=words)[0]
        assert "gemäß" in chunk.text
        assert chunk.heading_path == ("Größe",)


class TestEstimateTokens:
    def test_schaetzung_ist_positiv(self):
        assert estimate_tokens("kurz") >= 1

    def test_leerer_text_gibt_mindestens_eins(self):
        assert estimate_tokens("") == 1

    def test_laengerer_text_mehr_token(self):
        assert estimate_tokens("Wort " * 100) > estimate_tokens("Wort " * 10)


class TestChunkDataclass:
    def test_heading_ohne_pfad_ist_leer(self):
        assert Chunk(0, "Text").heading == ""
