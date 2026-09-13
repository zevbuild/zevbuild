# 🛡️ Zevbuild — Privacy-First Systems & Resilient Digital Products

> **Official Main Repository & Organization Portal for [Zevbuild Studio](https://github.com/zevbuild)**  
> Engineering zero-knowledge cryptographic vaults, offline-first peer-to-peer mesh protocols, real-time transit systems, and statistical engines.

[![Website](https://img.shields.io/badge/Website-zevbuild.github.io-8b5cf6?style=for-the-badge&logo=googlechrome)](https://zevbuild.github.io)
[![Organization](https://img.shields.io/badge/GitHub-zevbuild-10b981?style=for-the-badge&logo=github)](https://github.com/zevbuild)
[![License: MIT](https://img.shields.io/badge/License-MIT-06b6d4?style=for-the-badge)](LICENSE)
[![Zero Knowledge](https://img.shields.io/badge/Security-Zero--Knowledge-ef4444?style=for-the-badge)](#core-architecture--privacy-manifesto)
[![Offline First](https://img.shields.io/badge/Architecture-Offline--First-f59e0b?style=for-the-badge)](#core-architecture--privacy-manifesto)

---

## ✨ About Zevbuild

**Zevbuild** is a multi-domain software engineering studio dedicated to building autonomous, sovereign, and resilient digital tools. We believe user data must remain in the user's hands — requiring zero-knowledge cryptography, peer-to-peer local-first architectures, and zero telemetry by default.

---

## 🚀 Flagship Products & Ecosystem

| Product | Domain | Tech Stack | Status / Links |
| :--- | :--- | :--- | :--- |
| **[ZevSafe](tools/zevsafe/)** | Zero-Knowledge Folder & Vault Encryption | WebCrypto API · AES-256-GCM · PBKDF2-SHA512 · PWA | [Live Demo](https://zevsafe.pages.dev) · [Source](https://github.com/zevbuild/zevsafe) |
| **[ZevSync](tools/zevsync/)** | Offline Bluetooth P2P Mesh & CAS Vault | Kotlin 2.0+ · Jetpack Compose M3 · Vector Clocks · CAS | [Releases / APK](https://github.com/zevbuild/zevsync/releases) · [Source](https://github.com/zevbuild/zevsync) |
| **[CollegeBus](tools/collegebus/)** | Real-Time Transit & Multi-Hop Schedule Tracker | Vanilla JS · PWA · Service Worker · Edge CDN | [Live Demo](https://collegebus.pages.dev) · [Source](https://github.com/zevbuild/collegebus) |
| **[Predictive Analytics](tools/satta-matka-tools/)** | Kalyan Matka Probabilistic Modeling Engine | Python 3.14 · Markov Chains · Realtime Sync Web Engine | [Web Dashboard](tools/satta-matka-tools/index.html) · [CLI Engine](tools/satta-matka-tools/main.py) |
| **Custom Engineering** | Bespoke Secure Systems & Client-First Apps | Full-Stack Web · Cryptographic Architecture | Inquire via [Email](mailto:zevbuildstudio@gmail.com) |

---

## 🔒 Core Architecture & Privacy Manifesto

1. **Zero-Knowledge Core (Client-Side Only)**  
   No plaintext files or master passwords ever leave your client device. Key derivation (PBKDF2 with 600,000 iterations) and symmetric encryption (AES-256-GCM) execute natively inside the browser or mobile runtime.
2. **Offline-First Resilience**  
   Designed to operate in air-gapped environments, disaster recovery scenarios, or offline situations via local caching and Bluetooth mesh communication.
3. **Zero Telemetry & Tracking**  
   No third-party trackers, analytics beacons, or behavioral monitoring. We do not collect or sell user metadata.
4. **Open Cryptographic Standards**  
   Strict adherence to NIST-approved primitives and peer-reviewed protocols (AES-GCM, SHA-512, Lamport Timestamps, Content-Addressable Storage).

---

## 🛠️ Repository Structure

```
zevbuild/
├── index.html                    # Official company main portal & ecosystem showcase
├── .gitignore                    # Git hygiene rules
├── README.md                     # Organization profile and architecture documentation
└── tools/                        # Project suites and tool implementations
    ├── zevsafe/                  # Zero-knowledge in-browser folder encryption portal
    ├── zevsync/                  # Offline Bluetooth mesh synchronizer (Android)
    ├── collegebus/               # Student transit tracker & route planner PWA
    └── satta-matka-tools/        # Kalyan historical predictive modeling engine & dashboard
```

---

## 🌐 Local Development & Serving

To view and run the company main portal locally:

```bash
# Clone the repository
git clone https://github.com/zevbuild/zevbuild.git
cd zevbuild

# Run any static HTTP server (e.g. Python)
python -m http.server 8000

# Open in your browser:
# http://localhost:8000
```

---

## 📬 Contact & Inquiries

- **Organization**: [github.com/zevbuild](https://github.com/zevbuild)
- **Email**: [zevbuildstudio@gmail.com](mailto:zevbuildstudio@gmail.com)

---

## 📄 License
All proprietary and open-source assets under Zevbuild are released under the [MIT License](LICENSE).  
Copyright © 2026 Zevbuild Studio. All rights reserved.
