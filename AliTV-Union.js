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
        <html>
        <body>
            <p>Redirecting to authentication status check...</p>
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
            <html>
            <body>
                <div>
                    <p>Refresh Token: </p>
                    <p>${refreshToken}</p>
                    
                    <p>-----------------------------------------------------------------</p>
                    <p><a href="https://github.com/ACGDatabase/ACGDB/blob/main/AliTV-Union.js" target="_blank">Source Code</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://t.me/acgdb_channel/71" target="_blank">Original Post</a></p>
                    <p><strong>Welcome to <a href="https://acgdb.de" target="_blank">ACG Database</a>, where all ACG resources meet. </strong></p>
                </div>
            </body>
            </html>
        `
        return new Response(html, { headers: { 'Content-Type': 'text/html' } })
    } else {
        const qrLink = "https://openapi.alipan.com/oauth/qrcode/"+ qrID

        let html = `
            <html>
            <body>
                <div>
                    <img src="${qrLink}" alt="QR Code"/>
                    <p>Or login using <a href="https://www.aliyundrive.com/o/oauth/authorize?sid=${sid}" target="_blank">this link</a></p>
                    <p>Waiting for authentication...</p>

                    <p>-----------------------------------------------------------------</p>
                    <p><a href="https://github.com/ACGDatabase/ACGDB/blob/main/AliTV-Union.js" target="_blank">Source Code</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://t.me/acgdb_channel/71" target="_blank">Original Post</a></p>
                    <p><strong>Welcome to <a href="https://acgdb.de" target="_blank">ACG Database</a>, where all ACG resources meet. </strong></p>
                </div>
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
