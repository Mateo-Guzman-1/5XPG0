# deploy/: the "yes" window-model release

This folder is the first release, frozen: a dense window model for the keyword
**"yes"** (`manifest.json`: `"keyword": "yes"`), with the bitstreams and firmware
that run it on the PYNQ-Z2 (`keyword.bit` / `keyword_kdot.bit`, ABI 2).

The project's keyword is now **"sheila"**, detected against everything else by a
streaming SNN (`../README.md`, `../JOURNAL.md` entries 22 onwards). Its integer
model is `../results/models/sheila_stream_int8.npz`; it replaces this release
once it has passed the board test. Nothing in this folder detects "sheila".
