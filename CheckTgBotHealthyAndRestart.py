import asyncio
import subprocess
import time
from telethon import TelegramClient, events

api_id = ''
api_hash = ''
command = ''
bot_username = ''
bot_id = ''
process_path = ''

client = TelegramClient('userbot', api_id, api_hash)
process = None
last_response_time = time.time() - 10
healthy_status = True
check_interval = 30
healthy_until = time.time() - check_interval
async def start_process():
    global process
    if process:
        process.kill()
    process = subprocess.Popen(['python3', process_path])
    print('Started process')
    await client.send_message(bot_username, command)
    print('Sent command by restarting process')

async def send_command():
    global last_response_time, healthy_status
    while True:
        if healthy_status:
            await client.send_message(bot_username, command)
            print('Sent command')
        else:
            print('Bot is unhealthy, skip sending command')
        await asyncio.sleep(check_interval)  # 5 minutes

@client.on(events.NewMessage)
async def my_event_handler(event):
    global last_response_time,healthy_until
    if str(event.sender_id) == bot_id:
        healthy_until = time.time() + check_interval + 10
        print('Received response')
        last_response_time = time.time()

async def check_response():
    global last_response_time, healthy_status
    asyncio.ensure_future(send_command())
    while True:
        if time.time() > healthy_until:
            print('Bot is unhealthy, restarting')
            await start_process()
            healthy_status = False
        else:
            healthy_status = True
        await asyncio.sleep(1)  # check every second

async def main():
    await start_process()
    await check_response()

client.start(phone='YOURPHONE')
print('Client started')
client.loop.run_until_complete(main())

