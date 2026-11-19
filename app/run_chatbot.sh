#!/bin/sh
# ALWAYS run your_chat_bot from THIS directory: the TuyaOpen KV store that holds
# the device activation data lives at ./tuyadb/tuyadb, i.e. relative to the cwd.
# Running the elf from dist/ silently starts an UNACTIVATED device.
#
# ---------------------------------------------------------------------------
# Terminal (termios) guard
# ---------------------------------------------------------------------------
# tkl_uart.c:tkl_uart_init() puts stdin into  -ICANON -ECHO  mode for the CLI
# but keeps the original settings in a LOCAL variable and never writes them
# back, and tkl_uart_deinit() does not restore them either. If the app dies
# abnormally (SIGSEGV / SIGABRT / killed), the shell is left with ECHO off:
# typed characters are invisible while commands still run on Enter.
#
# Save the tty state here and ALWAYS give it back after the app exits, no
# matter how it exited. (NOTE: no `exec` -- exec would replace this shell and
# the restore code would never run.)
cd "$(dirname "$0")" || exit 1

STTY_SAVED=$(stty -g 2>/dev/null)

restore_tty() {
    if [ -n "$STTY_SAVED" ]; then
        stty "$STTY_SAVED" 2>/dev/null
    else
        stty sane 2>/dev/null
    fi
}
trap 'restore_tty' EXIT INT TERM HUP

./dist/your_chat_bot_1.0.1/your_chat_bot_1.0.1.elf "$@"
RC=$?

# Belt and braces: also restore on the straight-line path (the EXIT trap is
# enough in practice, but this keeps working if the trap is reset by the app).
restore_tty

exit $RC
