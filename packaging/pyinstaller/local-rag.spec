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

from PyInstaller.utils.hooks import collect_all, collect_submodules

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

# Projekt-Konfiguration muss ins Bundle (Modelle/Geräte pro Plattformklasse).
datas += [("config/platforms.toml", "config")]

block_cipher = None

# Läuft aus dem Repo-Root: `pyinstaller packaging/pyinstaller/local-rag.spec`.
a = Analysis(
    ["packaging/pyinstaller/app.py"],
    pathex=["."],
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
