- **`mime serve` says the `ssh` command that reaches it from another machine**, after its ready line: each port
  forwarded to the address it is bound to, as `ssh -N -L <port>:127.0.0.1:<port> -L <port>:127.0.0.1:<port>
  <user>@<host>` (`mime.live._serve.tunnel`); `docs/live.md` shows it.
