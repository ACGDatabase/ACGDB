addEventListener('fetch', event => {
    event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
    const url = new URL(request.url)

    if (url.pathname === '/auth') {
        return handleAuthRequest()
    } else if (url.pathname === '/check-status') {
        return handleStatusRequest(url)
    } else if (url.pathname === '/token') {
        return handleTokenRequest(request)
    } else {
        return new Response('Not Found', { status: 404 })
    }
}

async function handleAuthRequest() {
    const apiResponse = await fetch('http://api.extscreen.com/aliyundrive/qrcode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
            'scopes': 'user:base,file:all:read,file:all:write',
            'width': 500,
            'height': 500
        })
    })

    const data = await apiResponse.json()
    const qrData = data.data
    const sid = qrData.sid

    const responseHeaders = new Headers()
    const qrLink = qrData.qrCodeUrl
    const qrID = qrLink.split('/qrcode/')[1]
    responseHeaders.set('Content-Type', 'text/html')
    responseHeaders.set('Refresh', '0; url=/check-status?sid=' + sid + "&qrid=" + qrID)
    responseHeaders.set('Cache-Control', 'no-cache, no-store, must-revalidate')
    const html = `
        <!DOCTYPE html>
        <html>
        <head>
            <link href="https://cdn.jsdelivr.net/npm/bulma@0.9.3/css/bulma.min.css" rel="stylesheet">
            <title>Redirecting</title>
            <style>
                body {
                    background-color: #f5f5f5;
                }
                .hero-body {
                    text-align: center;
                }
            </style>
        </head>
        <body>
            <section class="hero is-info is-fullheight">
                <div class="hero-body">
                    <div class="container">
                        <h1 class="title">
                            Redirecting
                        </h1>
                        <h2 class="subtitle">
                            Redirecting to authentication status check...
                        </h2>
                        <button class="button is-loading">Loading</button>
                    </div>
                </div>
            </section>
        </body>
        </html>
    `
    return new Response(html, { headers: responseHeaders })
}

async function handleStatusRequest(url) {
    const sid = url.searchParams.get('sid')
    const qrID = url.searchParams.get('qrid')
    const statusResponse = await fetch(`https://openapi.alipan.com/oauth/qrcode/${sid}/status`)
    const statusData = await statusResponse.json()
    const status = statusData.status

    if (status === 'LoginSuccess') {
        const authCode = statusData.authCode
        const tokenResponse = await fetch('http://api.extscreen.com/aliyundrive/token', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ 'code': authCode })
        })
        const tokenData = await tokenResponse.json()
        const refreshToken = tokenData.data.refresh_token

        let html = `
            <!DOCTYPE html>
            <html>
            <head>
                <link href="https://cdn.jsdelivr.net/npm/bulma@0.9.3/css/bulma.min.css" rel="stylesheet">
                <title>Login Success</title>
                <style>
                    body {
                        background-color: #f5f5f5;
                    }
                    .container {
                        margin-top: 50px;
                    }
                    .box {
                        padding: 20px;
                    }
                </style>
            </head>
            <body>
                <section class="section">
                    <div class="container">
                        <div class="box">
                            <h5 class="title is-5">Refresh Token</h5>
                            <div class="field has-addons">
                                <div class="control is-expanded">
                                    <input id="refreshToken" class="input" type="text" value="${refreshToken}" readonly>
                                </div>
                                <div class="control">
                                    <button id="copyButton" class="button is-info">
                                        Copy
                                    </button>
                                </div>
                            </div>
                            <hr>
                            <p><a href="https://github.com/ACGDatabase/ACGDB/blob/main/AliTV-Union.js" target="_blank">Source Code</a></p>
                            <p><a href="https://t.me/acgdb_channel/71" target="_blank">Original Post</a></p>
                            <p><strong>Welcome to <a href="https://acgdb.de" target="_blank">ACG Database</a>, where all ACG resources meet.</strong></p>
                        </div>
                    </div>
                </section>
                <script>
                    document.getElementById('copyButton').addEventListener('click', function() {
                        const copyText = document.getElementById('refreshToken');
                        copyText.select();
                        copyText.setSelectionRange(0, 99999); // For mobile devices
                        document.execCommand('copy');
                    });
                </script>
            </body>
            </html>
        `
        return new Response(html, { headers: { 'Content-Type': 'text/html' } })
    } else {
        const qrLink = "https://openapi.alipan.com/oauth/qrcode/"+ qrID

        let html = `
            <!DOCTYPE html>
            <html>
            <head>
                <link href="https://cdn.jsdelivr.net/npm/bulma@0.9.3/css/bulma.min.css" rel="stylesheet">
                <title>Waiting for Authentication</title>
                <style>
                    body {
                        background-color: #f5f5f5;
                    }
                    .container {
                        margin-top: 50px;
                        text-align: center;
                    }
                    .box {
                        padding: 20px;
                    }
                </style>
            </head>
            <body>
                <section class="section">
                    <div class="container">
                        <div class="box">
                            <figure class="image is-128x128 is-inline-block">
                                <img src="${qrLink}" alt="QR Code"/>
                            </figure>
                            <p class="mt-3">Or login using <a href="https://www.aliyundrive.com/o/oauth/authorize?sid=${sid}" target="_blank">this link</a></p>
                            <p class="is-size-6 has-text-grey">Waiting for authentication...</p>
                            <hr>
                            <p><a href="https://github.com/ACGDatabase/ACGDB/blob/main/AliTV-Union.js" target="_blank">Source Code</a></p>
                            <p><a href="https://t.me/acgdb_channel/71" target="_blank">Original Post</a></p>
                            <p><strong>Welcome to <a href="https://acgdb.de" target="_blank">ACG Database</a>, where all ACG resources meet.</strong></p>
                        </div>
                    </div>
                </section>
            </body>
            </html>
        `
        const responseHeaders = new Headers()
        responseHeaders.set('Content-Type', 'text/html')
        responseHeaders.set('Refresh', '10; url=/check-status?sid=' + sid + "&qrid=" + qrID)
        return new Response(html, { headers: responseHeaders })
    }
}

async function handleTokenRequest(request) {
    const originalUrl = "https://api.nn.ci/alist/ali_open/token"
    const { headers } = request
    const body = await request.json()

    const clientId = body.client_id
    const clientSecret = body.client_secret
    const grantType = body.grant_type
    const refreshToken = body.refresh_token

    if (clientId && clientSecret) {
        const response = await fetch(originalUrl, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(body)
        })
        const data = await response.json()
        return new Response(JSON.stringify(data), { status: response.status, headers: { 'Content-Type': 'application/json' } })
    }

    let decodedToken
    try {
        decodedToken = JSON.parse(atob(refreshToken.split('.')[1]))
    } catch (e) {
        return new Response(JSON.stringify({ error: "Invalid token" }), { status: 400, headers: { 'Content-Type': 'application/json' } })
    }

    if (decodedToken.aud !== '6b5b52e144f748f78b3f96a2626ed5d7') {
        const response = await fetch(originalUrl, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(body)
        })
        const data = await response.json()
        return new Response(JSON.stringify(data), { status: response.status, headers: { 'Content-Type': 'application/json' } })
    }

    const tokenInfoResponse = await fetch('http://api.extscreen.com/aliyundrive/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ 'refresh_token': refreshToken })
    })

    if (tokenInfoResponse.status !== 200) {
        return new Response(JSON.stringify({ error: "Failed to fetch token info" }), { status: 500, headers: { 'Content-Type': 'application/json' } })
    }

    const tokenInfo = await tokenInfoResponse.json().then(data => data.data)
    const accessToken = tokenInfo.access_token
    const newRefreshToken = tokenInfo.refresh_token

    return new Response(JSON.stringify({
        token_type: "Bearer",
        access_token: accessToken,
        refresh_token: newRefreshToken,
        expires_in: 7200
    }), { headers: { 'Content-Type': 'application/json' } })
}
