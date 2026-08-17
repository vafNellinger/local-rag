# Installer & Signierung

Der CI-Workflow (`.github/workflows/build.yml`) baut je Betriebssystem einen
per Wizard installierbaren Installer:

| OS      | Installer            | Baut aus                    | Installationsziel                        |
|---------|----------------------|-----------------------------|------------------------------------------|
| Linux   | `.deb`               | `dist/local-rag/`           | `/opt/local-rag`, Menü-Eintrag           |
| Windows | NSIS `…-setup.exe`   | `dist/local-rag/`           | `%LOCALAPPDATA%\Programs\local-rag`       |
| macOS   | `.dmg`               | `dist/local-rag.app`        | Ziehen nach `/Programme`                  |

Ohne Signierung funktionieren alle drei — aber das Betriebssystem warnt beim
ersten Start. Die Signierung ist **optional** und schaltet sich automatisch ein,
sobald die jeweiligen GitHub-Secrets hinterlegt sind. Fehlen sie, überspringt
der Build die Signierung mit einer Notiz und liefert den unsignierten Installer.

Secrets hinterlegen: GitHub → Repo → **Settings › Secrets and variables ›
Actions › New repository secret**.

---

## Windows (SmartScreen)

Ohne Signatur zeigt Windows SmartScreen beim Start „Der Computer wurde durch
Windows geschützt". Der Nutzer kann über *Weitere Informationen › Trotzdem
ausführen* fortfahren. Zum Beseitigen der Warnung braucht es ein
**Code-Signing-Zertifikat** (OV oder EV) einer anerkannten CA (z. B. DigiCert,
Sectigo, ~250–500 €/Jahr). EV-Zertifikate bauen SmartScreen-Reputation sofort
auf, OV erst nach einigen Downloads.

Das Zertifikat als `.pfx`/`.p12` exportieren und als Base64 ablegen:

```bash
base64 -w0 zertifikat.pfx > cert.b64   # Inhalt kopieren
```

Secrets:

| Secret                    | Inhalt                                   |
|---------------------------|------------------------------------------|
| `WINDOWS_CERT_BASE64`     | Base64 der `.pfx`-Datei                   |
| `WINDOWS_CERT_PASSWORD`   | Passwort des `.pfx`-Exports              |

Der Workflow signiert dann `…-setup.exe` mit `signtool` samt Zeitstempel.

---

## macOS (Gatekeeper & Notarisierung)

Ohne Signatur blockiert Gatekeeper die App; der Nutzer muss *Rechtsklick ›
Öffnen* wählen. Für den reibungslosen Doppelklick sind **zwei** Schritte nötig,
beide brauchen einen **Apple Developer Account (99 $/Jahr)**:

1. **Signieren** mit einem *Developer ID Application*-Zertifikat.
2. **Notarisieren** – Apple prüft das `.dmg` und hängt ein Ticket an (`staple`).

Zertifikat aus der Schlüsselbundverwaltung als `.p12` exportieren, Base64:

```bash
base64 -i developer-id.p12 -o cert.b64
```

Für die Notarisierung ein **app-spezifisches Passwort** unter
<https://appleid.apple.com> anlegen (nicht das Apple-ID-Login-Passwort).

Secrets:

| Secret                     | Inhalt                                                    |
|----------------------------|-----------------------------------------------------------|
| `MACOS_CERT_P12_BASE64`    | Base64 der `.p12`-Datei                                    |
| `MACOS_CERT_PASSWORD`      | Passwort des `.p12`-Exports                                |
| `MACOS_SIGN_IDENTITY`      | z. B. `Developer ID Application: Name (TEAMID)`            |
| `MACOS_NOTARY_APPLE_ID`    | Apple-ID (E-Mail)                                          |
| `MACOS_NOTARY_PASSWORD`    | app-spezifisches Passwort                                  |
| `MACOS_NOTARY_TEAM_ID`     | 10-stellige Team-ID (im Developer-Portal)                 |

Der Workflow signiert das `.app` vor dem `.dmg`-Bau (Hardened Runtime) und
notarisiert das fertige `.dmg`.

---

## Linux (.deb)

Ein direkt heruntergeladenes `.deb` braucht keine Signatur — `dpkg -i` /
`apt install ./local-rag_*.deb` installiert es ohne Warnung. Signiert wird auf
Debian erst, wenn das Paket über ein **APT-Repository** verteilt wird; dann
signiert man das *Repository* (nicht das Paket) mit einem GPG-Schlüssel
(`Release.gpg` / `InRelease`). Das ist erst relevant, wenn wir einen eigenen
apt-Kanal aufsetzen — für den Direkt-Download ist nichts zu tun.

Die WebKit-Laufzeit fürs native Fenster ist als `Depends` deklariert; `apt`
zieht sie beim Installieren nach. Fehlt sie (bei `dpkg -i` ohne
Abhängigkeitsauflösung), öffnet die Oberfläche im Browser statt im eigenen
Fenster.
