import csv
import requests
tenant_id="Your Tenant ID"
client_credentials = [
    ("AppID","App Secret"),
]
    
# function to get the access and refresh tokens
def get_tokens(client_id, client_secret, tenant_id, username, password):
    #get refresh token and access token
    url = "https://login.microsoftonline.com/{}/oauth2/v2.0/token".format(tenant_id)
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "password",
        #offline_access is required to get a refresh token
        "scope": "https://graph.microsoft.com/.default offline_access",
        "username": username,
        "password": password
    }
    response = requests.post(url, data=data)
    return response.json()["access_token"], response.json()["refresh_token"]
# function to get the drive id
def get_drive_id(access_token):
    # construct the drive request url
    url = "https://graph.microsoft.com/v1.0/me/drive"

    # construct the drive request headers
    headers = {
        "Authorization": "Bearer " + access_token
    }

    # send the drive request and get the response
    response = requests.get(url, headers=headers)
    print("Response: "+response.text)
    # return the drive id
    return response.json()["id"]
def get_full_name(access_token):
    url="https://graph.microsoft.com/v1.0/me"
    headers = {
        "Authorization": "Bearer " + access_token
    }
    response = requests.get(url, headers=headers)
    return response.json()["displayName"]
def get_client_id_secret(now_row):
    index = now_row % len(client_credentials)
    client_id = client_credentials[index][0]
    client_secret = client_credentials[index][1]
    print("Now using:",client_id,"now_row:",now_row)
    return client_id, client_secret
# read the csv file containing the office 365 accounts and passwords
with open("accounts.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    # next(reader) # skip the header row
    now_row=0
    # open the rclone.conf file for appending
    with open("rclone.conf", "a") as conf:
        for row in reader:
            username = row[0]
            password = row[1]
            client_id, client_secret = get_client_id_secret(now_row)
            # get the access and refresh tokens
            access_token, refresh_token = get_tokens(client_id, client_secret, tenant_id, username, password)

            # get the drive id
            drive_id = get_drive_id(access_token)
            #get full name

            #get expiry
            #username=get_full_name(access_token)
            #username: a@c.com-> a
            username="ent"+username.split("@")[0]
            print("username:",username)
            # construct the rclone config entry
            config_entry = "[{}]\ntype=onedrive\nclient_id={}\nclient_secret={}\ntoken={{\"access_token\":\"{}\",\"token_type\":\"Bearer\",\"refresh_token\":\"{}\",\"expiry\":\"2022-11-05T14:43:09.1206112+08:00\"}}\ndrive_id={}\ndrive_type=business\n".format(username, client_id, client_secret, access_token, refresh_token, drive_id)
            # append the config entry to the rclone.conf file
            conf.write(config_entry)
            now_row+=1
