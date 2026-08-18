# Agent Notes

## NEMO2 Access

- Do not blindly retry SSH to NEMO2 after authentication failures.
- If SSH reports `Permission denied`, `ssh_askpass`, `Too many authentication failures`, or appears to require keyboard-interactive auth, assume the user may need to manually log in with OTP first.
- Ask the user to complete a manual login from their terminal, for example `ssh nemo2`, before retrying automated SSH checks.
- When retrying after user confirmation, prefer an explicit key/config-minimal command if needed, e.g. `ssh -F /dev/null -o IdentityAgent=none -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519_NEMO2_v2 fr_xl1014@nemo2-login.nemo.uni-freiburg.de`.
