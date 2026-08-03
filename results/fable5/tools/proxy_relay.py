"""Temporary TCP relay: 0.0.0.0:7891 -> 127.0.0.1:7890 (ClashX).

Lets the colima VM / Docker containers reach the host-only ClashX proxy so the
verifier's Codex audit can call the OpenAI API. Kill this process after the
benchmark verification finishes.
"""
import asyncio

LISTEN_PORT = 7891
TARGET_HOST, TARGET_PORT = "127.0.0.1", 7890


async def pipe(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(client_reader, client_writer):
    try:
        remote_reader, remote_writer = await asyncio.open_connection(TARGET_HOST, TARGET_PORT)
    except OSError:
        client_writer.close()
        return
    await asyncio.gather(
        pipe(client_reader, remote_writer),
        pipe(remote_reader, client_writer),
    )


async def main():
    server = await asyncio.start_server(handle, "0.0.0.0", LISTEN_PORT)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
