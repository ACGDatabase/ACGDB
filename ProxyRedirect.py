import requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import asyncio
import flask
import random
import threading
from waitress import serve
import time

#storage url and status
proxies=[
    {"url":"https://example1.com","status":True},
    {"url":"https://example2.com","status":True},
    
]
def check_alive(url):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return True
        else:
            return False
    except:
        return False

app=flask.Flask(__name__)
# route all requests to flask here
# 302 to a random proxy address
# for example: http://localhost:5000/1/b/c?d=e -> http://live_proxy.com/1/b/c?d=e
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    proxy=None
    #get random proxy in status=True
    proxy = random.choice([proxy_loop for proxy_loop in proxies if proxy_loop["status"]==True])
    #check proxy status
    if proxy!=None:
        #preserve query string
        query_string=flask.request.query_string.decode("utf-8")
        #return flask.redirect(proxy["url"]+path, code=302)
        if query_string=="":
            return flask.redirect(proxy["url"]+'/'+path, code=302)
        else:
            return flask.redirect(proxy["url"]+'/'+path+'?'+query_string, code=302)
    else:
        #output in html: no proxy avaliable
        return flask.Response("No proxy avaliable", mimetype='text/html')


def main():
    print("Count: "+str(len(proxies)))
    for proxy in proxies:
        if check_alive(proxy["url"])==False:
            proxy["status"]=False
            print(str(proxies.index(proxy)+1)+": "+proxy["url"]+" is down")
        else:
            proxy["status"]=True
            print(str(proxies.index(proxy)+1)+": "+proxy["url"]+" is up")
        if proxy["url"].endswith("eu.org"):
            time.sleep(0.2)
    print("Alive: "+str(len([proxy for proxy in proxies if proxy["status"]==True])))
main()
scheduler = AsyncIOScheduler()
#check proxy every 5 minute
scheduler.add_job(main, CronTrigger.from_crontab('*/5 * * * *'),misfire_grace_time=60,coalesce=True,max_instances=1)
scheduler.start()
def serve_in_thread():
    serve(app, host='0.0.0.0',port=5000)
thread = threading.Thread(target=serve_in_thread)
thread.start()

loop = asyncio.get_event_loop()
loop.run_forever()