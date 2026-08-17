# PyInstaller-Spec für local-rag — ein Bundle je Betriebssystem (onedir).
#
# Kein onefile: die Bundles wiegen mehrere Gigabyte (PyTorch), ein onefile würde
# sie bei jedem Start in ein Temp-Verzeichnis auspacken — langsam und
# fehleranfällig. Modelle werden NICHT gebündelt; sie kommen beim ersten Start
# in den user-cache (siehe rag/paths.py).
#
# Erster Wurf: collect_all() zieht die dynamischen Importe und Data-Files der
# fummeligen Pakete automatisch ein. Was der erste CI-Lauf noch vermisst, wird
# hier ergänzt.

import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

# PyInstaller löst Pfade relativ zum Spec-Verzeichnis auf, nicht zum CWD —
# deshalb alles absolut über SPECPATH, damit der Aufruf von überall klappt.
_HERE = os.path.abspath(SPECPATH)  # packaging/pyinstaller
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))  # Repo-Wurzel

datas: list = []
binaries: list = []
hiddenimports: list = []

# Pakete, deren PyInstaller-Erkennung erfahrungsgemäß unvollständig ist: Modelle,
# Templates, native Bibliotheken, lazy Importe. Vollständig einsammeln.
for pkg in (
    "nicegui",
    "docling",
    "docling_core",
    "docling_ibm_models",
    "docling_parse",
    "easyocr",
    "sentence_transformers",
    "onnxruntime",
    "llama_cpp",
    "webview",  # pywebview
    "sqlite_vec",
):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # Paket fehlt auf dieser Plattform → überspringen
        print(f"collect_all({pkg}) übersprungen: {exc}")

# torch/transformers haben brauchbare Built-in-Hooks; nur die dynamisch
# geladenen Submodule sicherheitshalber ergänzen.
hiddenimports += collect_submodules("transformers")

# pywebview öffnet auf Linux sein Fenster über GTK/WebKit — angesprochen via
# PyGObject (gi). Ohne das im Bundle fällt die Oberfläche auf den Browser
# zurück (siehe rag/ui._webview_verfuegbar). Der eingebaute gi-Hook zieht die
# GObject-Introspection-Typelibs; die konkret genutzten Namespaces müssen als
# hiddenimport bekannt sein, damit WebKit2 mitkommt. Die schweren nativen
# GTK/WebKit-Bibliotheken selbst werden NICHT gebündelt — sie kommen als
# System-Abhängigkeit über das .deb (Depends), das ist auf Debian robuster als
# der Versuch, den ganzen WebKit-Prozessbaum einzupacken.
if sys.platform.startswith("linux"):
    hiddenimports += [
        "gi",
        "gi.repository.GLib",
        "gi.repository.Gio",
        "gi.repository.GObject",
        "gi.repository.Gtk",
        "gi.repository.Gdk",
        "gi.repository.WebKit2",
        "gi.repository.Soup",
    ]
    try:
        d, b, h = collect_all("gi")
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:
        print(f"collect_all(gi) übersprungen: {exc}")

# Projekt-Konfiguration muss ins Bundle (Modelle/Geräte pro Plattformklasse).
datas += [(os.path.join(_ROOT, "config", "platforms.toml"), "config")]

block_cipher = None

a = Analysis(
    [os.path.join(_HERE, "app.py")],
    pathex=[_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "pytest"],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="local-rag",
    debug=False,
    strip=False,
    upx=False,
    console=False,  # GUI-App: kein Konsolenfenster
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="local-rag",
)

# Auf macOS wird aus dem Verzeichnis-Bundle zusätzlich ein .app-Paket — nur so
# lässt es sich per Doppelklick starten und in ein .dmg legen. Der Launcher
# öffnet ein natives Fenster (WKWebView), deshalb keine Menüleiste erzwingen.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="local-rag.app",
        icon=None,
        bundle_identifier="de.vonaffenfels.local-rag",
        info_plist={
            "CFBundleName": "local-rag",
            "CFBundleDisplayName": "local-rag",
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleVersion": "0.1.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
            # Ein reines Fenster-Programm, kein Dock-loses Hintergrundtool.
            "LSBackgroundOnly": False,
        },
    )
