- **`mime serve` binds loopback only unless `--allow-remote-commands`** (it replaces `--remote`, and `serve()`'s
  `allow_remote_commands=` replaces `remote=`). A live session has no authentication, so an address that is not
  loopback (127.0.0.0/8, `::1`, or `localhost` where it resolves to loopback only; a host name is not resolved for
  this) is refused before anything binds: `live.commands-not-loopback` for the command socket,
  `live.frames-not-loopback` for the frame socket. The flag binds it all the same and warns on standard error, naming
  each such address and what it gives whoever reaches it. `localhost` was trusted by its name before; it is now checked
  by what it resolves to. A port in use, by a second `mime serve` say, is refused naming that port and its option
  (`live.address-in-use`), with free ports (`tcp://127.0.0.1:0`) as the way out.
