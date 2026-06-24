# Export Licensing Checklist

Use this checklist for the first public binary release. It is a release aid,
not legal advice.

- [ ] Confirm the copyright owner text in `COPYRIGHT` is correct.
- [ ] Include the GPLv3 license, `COPYRIGHT`, and corresponding Color Track
      source/build scripts with every distributed binary release.
- [ ] Create a fresh dependency inventory from the exact virtual environment
      used by the installer build.
- [ ] Copy the full license/notice files for every bundled Python wheel and
      native DLL into the installer or an included `licenses` directory.
- [ ] Record the exact FFmpeg version and complete `-buildconf` output used by
      the release.
- [ ] Make the matching FFmpeg corresponding source and license notices
      available to every recipient of a release containing FFmpeg DLLs.
- [ ] Do not bundle CUDA, NVIDIA Video Codec SDK, or GPU-driver material unless
      the relevant NVIDIA terms allow that specific redistribution.
- [ ] Review the final installer contents and notices before publication.

`THIRD_PARTY_NOTICES.md` identifies the currently declared project
dependencies and the two decisions that need special attention.
