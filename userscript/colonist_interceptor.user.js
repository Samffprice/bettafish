// ==UserScript==
// @name             AICatan Colonist Interceptor
// @description      Intercepts colonist.io WebSocket messages and forwards them to the AICatan bridge bot.
// @include          /^https://colonist\.io\/.*/
// @run-at           document-start
// @grant            none
// @require          https://cdn.jsdelivr.net/npm/msgpack-lite@0.1.26/dist/msgpack.min.js
// @version          3.0.0
// ==/UserScript==

// --- Action constants (colonist.io client→server action IDs) ---
// ALL VERIFIED from hex dumps. Build flow: SELECT(index) → SELECT(null) → BUILD_X(index)
// Setup build IDs are N, normal play IDs are N+1 (roads 11/12, settlements 15/16)
const colonistioActions = Object.freeze({
    // Building: SELECT(index) → SELECT(null) → BUILD_X(index)
    SELECT: 66,              // VERIFIED
    BUILD_ROAD_SETUP: 11,    // VERIFIED: setup +1 = normal
    BUILD_ROAD: 12,          // VERIFIED
    BUILD_SETTLEMENT_SETUP: 15, // VERIFIED: setup +1 = normal
    BUILD_SETTLEMENT: 16,    // VERIFIED: mid-game
    BUILD_CITY: 19,          // VERIFIED
    // Core
    THROW_DICE: 2,           // VERIFIED
    MOVE_ROBBER: 3,          // VERIFIED: payload=tileIndex
    END_TURN: 6,             // VERIFIED
    ROB_PLAYER: 5,            // VERIFIED: payload=playerColor
    CONFIRM_DISCARD: 7,       // VERIFIED: payload=array of resource IDs
    SELECT_CARDS: 8,          // VERIFIED: payload=array (incremental selection)
    BUY_DEVELOPMENT_CARD: 9,  // VERIFIED
    // Dev cards: SELECT_DEV_CARD → CONFIRM_DEV_CARD → (card-specific actions)
    CONFIRM_DEV_CARD: 48,     // VERIFIED: confirm playing selected dev card
    SELECT_DEV_CARD: 53,      // VERIFIED: select which dev card to play
    // Trade
    INIT_TRADE: 47,          // VERIFIED
    CREATE_TRADE: 49,        // VERIFIED
    TRADE_RESPONSE: 50,      // VERIFIED: response=0 accept, response=1 reject
    EXECUTE_TRADE: 51,       // VERIFIED: execute your own trade after someone accepts
    // Dev card specific build confirmations
    CONFIRM_BUILD_DEV: 21    // VERIFIED: confirm road placement during Road Building dev card
})

// Dev card type IDs (used as payload for CONFIRM_DEV_CARD action 48)
// Old IDs were offset by +4
const devCards = Object.freeze({
    KNIGHT: 11,          // VERIFIED
    VICTORY_POINT: 12,   // VERIFIED
    MONOPOLY: 13,        // predicted (+4 offset)
    ROAD_BUILDING: 14,   // predicted (+4 offset)
    YEAR_OF_PLENTY: 15   // predicted (+4 offset)
})

// --- Bot WebSocket with reconnect logic ---
const BOT_WS_URL = 'ws://localhost:8765'
const MAX_RECONNECT_DELAY_MS = 10000
const MIN_RECONNECT_DELAY_MS = 1000

let botSocket = null
let gameSocket = null
let gameSocketConfirmed = false
let reconnectDelay = MIN_RECONNECT_DELAY_MS
let reconnectTimer = null
let messageBuffer = []
let myPlayerColor = 1
let lastType4Message = null  // Cached type 4 game init for reconnect resend

// --- Protocol state ---
// Client→server format: [0x03][0x01][serverIdLen][serverId bytes][msgpack {action, payload, sequence}]
let outgoingSequence = 0
let capturedServerId = null   // e.g. "0C0C0B", captured from real client sends
let isOurSend = false         // Skip spy capture on our own sends
let pendingSends = []         // Queue sends until serverId is captured

function connectBotSocket() {
    if (botSocket && (botSocket.readyState === WebSocket.CONNECTING || botSocket.readyState === WebSocket.OPEN)) {
        return
    }

    console.log('[AICatan] Connecting to bot server at', BOT_WS_URL)
    botSocket = new window._NativeWebSocket(BOT_WS_URL)

    botSocket.onopen = function () {
        console.log('[AICatan] Connected to bot server')
        document.title = '[BOT] ' + document.title.replace('[BOT] ', '')
        reconnectDelay = MIN_RECONNECT_DELAY_MS

        // Resend cached type 4 game init so bridge gets full state on reconnect
        if (lastType4Message) {
            console.log('[AICatan] Resending cached type 4 game init to bridge')
            try {
                botSocket.send(lastType4Message)
            } catch (e) {
                console.error('[AICatan] Failed to resend type 4:', e)
            }
        }

        while (messageBuffer.length > 0) {
            const msg = messageBuffer.shift()
            try {
                botSocket.send(msg)
            } catch (e) {
                console.error('[AICatan] Failed to flush buffered message:', e)
            }
        }
    }

    botSocket.onmessage = function (event) {
        try {
            handleBotMessage(event.data)
        } catch (e) {
            console.error('[AICatan] Error handling bot message:', e)
        }
    }

    botSocket.onclose = function (event) {
        console.warn('[AICatan] Bot server disconnected (code=' + event.code + '). Reconnecting in', reconnectDelay, 'ms')
        document.title = document.title.replace('[BOT] ', '[BOT-DC] ')
        scheduleReconnect()
    }

    botSocket.onerror = function (error) {
        console.error('[AICatan] Bot WebSocket error:', error)
    }
}

function scheduleReconnect() {
    if (reconnectTimer) {
        clearTimeout(reconnectTimer)
    }
    reconnectTimer = setTimeout(function () {
        reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY_MS)
        connectBotSocket()
    }, reconnectDelay)
}

function sendToBot(data) {
    const message = typeof data === 'string' ? data : JSON.stringify(data)
    if (botSocket && botSocket.readyState === WebSocket.OPEN) {
        botSocket.send(message)
    } else {
        if (messageBuffer.length < 100) {
            messageBuffer.push(message)
        }
        if (!botSocket || botSocket.readyState === WebSocket.CLOSED) {
            scheduleReconnect()
        }
    }
}

// --- Handle messages from the bot server ---
function handleBotMessage(rawData) {
    const parsedData = JSON.parse(rawData)
    if (parsedData.desc) {
        console.log('%c[AICatan] >> ' + parsedData.desc, 'color: #4CAF50; font-weight: bold')
    }
    console.log('[AICatan] Bot command:', parsedData)

    switch (parsedData.action) {
        case 0: { // Build road (setup/normal) — SELECT(edge) → SELECT(null) → BUILD_ROAD(edge)
            // Normal/setup roads need the SELECT flow; dev card roads (case 13) do NOT
            const roadAction = parsedData.setup
                ? colonistioActions.BUILD_ROAD_SETUP
                : colonistioActions.BUILD_ROAD
            sendGameAction(colonistioActions.SELECT, parsedData.data)
            sendGameAction(colonistioActions.SELECT, null)
            sendGameAction(roadAction, parsedData.data)
            break
        }
        case 1: { // Build settlement — SELECT(vertex) → SELECT(null) → BUILD_SETTLEMENT(vertex)
            const settlementAction = parsedData.setup
                ? colonistioActions.BUILD_SETTLEMENT_SETUP
                : colonistioActions.BUILD_SETTLEMENT
            sendGameAction(colonistioActions.SELECT, parsedData.data)
            sendGameAction(colonistioActions.SELECT, null)
            sendGameAction(settlementAction, parsedData.data)
            break
        }
        case 2: // Build city — SELECT(vertex) → SELECT(null) → BUILD_CITY(vertex=19)
            sendGameAction(colonistioActions.SELECT, parsedData.data)
            sendGameAction(colonistioActions.SELECT, null)
            sendGameAction(colonistioActions.BUILD_CITY, parsedData.data)
            break
        case 3: // Buy development card — action=9, payload=true
            sendGameAction(colonistioActions.BUY_DEVELOPMENT_CARD, true)
            break
        case 4: // Throw dice
            sendGameAction(colonistioActions.THROW_DICE, true)
            break
        case 5: // End turn
            sendGameAction(colonistioActions.END_TURN, true)
            break
        case 6: // Accept trade — same action as reject, response=0
            sendGameAction(colonistioActions.TRADE_RESPONSE, {
                id: parsedData.data.tradeId || parsedData.data,
                response: 0
            })
            break
        case 7: // Reject trade — same action as accept, response=1
            sendGameAction(colonistioActions.TRADE_RESPONSE, {
                id: parsedData.data.tradeId || parsedData.data,
                response: 1
            })
            break
        case 8: // Move robber — payload=tileIndex
            sendGameAction(colonistioActions.MOVE_ROBBER, parsedData.data)
            break
        case 9: // Rob player — action=5, payload=playerColor
            sendGameAction(colonistioActions.ROB_PLAYER, parsedData.data)
            break
        case 10: // Discard — select cards (8) then confirm (7)
            sendGameAction(colonistioActions.SELECT_CARDS, parsedData.data)
            sendGameAction(colonistioActions.CONFIRM_DISCARD, parsedData.data)
            break
        case 11: { // Create trade — send INIT_TRADE first, then CREATE_TRADE
            const isBankTrade = parsedData.data.bankTrade || false
            const creator = parsedData.data.creator != null ? parsedData.data.creator : myPlayerColor
            sendGameAction(colonistioActions.INIT_TRADE, true)
            sendGameAction(colonistioActions.CREATE_TRADE, {
                creator: creator,
                isBankTrade: isBankTrade,
                counterOfferInResponseToTradeId: null,
                offeredResources: parsedData.data.offered,
                wantedResources: parsedData.data.wanted
            })
            break
        }
        case 12: // Play development card — SELECT_DEV_CARD(48) → CONFIRM_DEV_CARD(cardTypeId)
            // Card type IDs: 11=Knight, 12=VP, 13=Monopoly, 14=RoadBuilding, 15=YearOfPlenty
            // After knight: server triggers MOVE_ROBBER + ROB_PLAYER via state diffs
            sendGameAction(colonistioActions.SELECT_DEV_CARD, 48)
            sendGameAction(colonistioActions.CONFIRM_DEV_CARD, parsedData.data)
            break
        case 13: // Build road during Road Building dev card — no SELECT step (unlike case 0)
            // Dev card roads: BUILD_ROAD_SETUP(edge) → CONFIRM_BUILD_DEV(edge)
            sendGameAction(colonistioActions.BUILD_ROAD_SETUP, parsedData.data)
            sendGameAction(colonistioActions.CONFIRM_BUILD_DEV, parsedData.data)
            break
        case 14: // Execute our trade with a specific acceptee
            sendGameAction(colonistioActions.EXECUTE_TRADE, {
                tradeId: parsedData.data.tradeId,
                playerToExecuteTradeWith: parsedData.data.playerColor
            })
            break
        default:
            console.warn('[AICatan] Unknown action code:', parsedData.action)
    }
}

// --- Send game action to colonist.io ---
// Builds the full binary frame: [0x03][0x01][sidLen][serverId][msgpack {action, payload, sequence}]
function sendGameAction(actionId, payload) {
    if (!gameSocket) {
        console.error('[AICatan] gameSocket not available yet')
        return
    }

    if (capturedServerId === null) {
        console.warn('[AICatan] ServerId not captured yet, queuing action=' + actionId)
        pendingSends.push({ actionId: actionId, payload: payload })
        return
    }

    _sendRaw(actionId, payload)
}

function _sendRaw(actionId, payload) {
    outgoingSequence++
    const msg = { action: actionId, payload: payload, sequence: outgoingSequence }

    try {
        // Encode msgpack payload — copy to avoid msgpack-lite's oversized internal buffer
        // standard msgpack.encode returns a Uint8Array view of a larger buffer
        const encoded = msgpack.encode(msg)
        const msgpackBytes = new Uint8Array(encoded.buffer, encoded.byteOffset, encoded.byteLength)

        // Build prefix: [0x03][0x01][len][serverId bytes]
        const serverIdBytes = new TextEncoder().encode(capturedServerId)
        const prefixLen = 3 + serverIdBytes.length

        // Combine into final frame
        const frame = new Uint8Array(prefixLen + msgpackBytes.length)
        frame[0] = 0x03
        frame[1] = 0x01
        frame[2] = serverIdBytes.length
        frame.set(serverIdBytes, 3)
        frame.set(msgpackBytes, prefixLen)

        const hex = Array.from(frame.slice(0, 50)).map(b => b.toString(16).padStart(2, '0')).join(' ')
        console.log('[AICatan] SENDING:', JSON.stringify(msg), 'hex(' + frame.length + 'b):', hex)

        isOurSend = true
        gameSocket.send(frame)
    } catch (e) {
        console.error('[AICatan] Failed to encode outgoing message:', e, msg)
    }
}

function flushPendingSends() {
    console.log('[AICatan] Flushing', pendingSends.length, 'pending sends')
    while (pendingSends.length > 0) {
        const queued = pendingSends.shift()
        _sendRaw(queued.actionId, queued.payload)
    }
}

// --- Intercept colonist.io WebSocket ---
window._NativeWebSocket = window.WebSocket

function patchWebSocket() {
    const _nativeProtoSend = window._NativeWebSocket.prototype.send
    window._NativeWebSocket.prototype.send = function (data) {
        try {
            const url = (this.url || '?')
            if (!url.includes('colonist.io')) {
                return _nativeProtoSend.call(this, data)
            }

            // Skip spy for our own sends
            if (isOurSend) {
                isOurSend = false
                return _nativeProtoSend.call(this, data)
            }

            if (data instanceof ArrayBuffer || (data && data.buffer instanceof ArrayBuffer)) {
                const raw = data instanceof ArrayBuffer ? data : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)
                const bytes = new Uint8Array(raw)

                // Log hex dump
                const hex = Array.from(bytes.slice(0, 50)).map(b => b.toString(16).padStart(2, '0')).join(' ')
                console.log('[AICatan] SEND hex (' + bytes.length + 'b):', hex)

                // Try to capture serverId from prefix: [type][0x01][len][serverId...]
                // type is 0x03 (game action) or 0x04 (heartbeat/system)
                if (bytes.length > 3 && (bytes[0] === 0x03 || bytes[0] === 0x04) && bytes[1] === 0x01) {
                    const sidLen = bytes[2]
                    if (sidLen > 0 && sidLen < 20 && bytes.length > 3 + sidLen) {
                        const sid = new TextDecoder().decode(bytes.slice(3, 3 + sidLen))

                        if (capturedServerId === null) {
                            capturedServerId = sid
                            console.log('[AICatan] CAPTURED serverId:', capturedServerId)
                            flushPendingSends()
                        }

                        // Decode msgpack payload after prefix for logging
                        const prefixEnd = 3 + sidLen
                        try {
                            const decoded = msgpack.decode(bytes.slice(prefixEnd))
                            console.log('[AICatan] SEND decoded:', JSON.stringify(decoded))

                            // Sync sequence counter
                            if (decoded && typeof decoded.sequence === 'number' && decoded.sequence >= outgoingSequence) {
                                outgoingSequence = decoded.sequence
                            }
                        } catch (e2) { /* ignore */ }
                    }
                }
            } else if (typeof data === 'string') {
                console.log('[AICatan] SEND text (' + data.length + '):', data.substring(0, 120))
            }
        } catch (e) { /* ignore */ }
        return _nativeProtoSend.call(this, data)
    }

    window.WebSocket = function (...args) {
        const socket = new window._NativeWebSocket(...args)
        const url = args[0] || ''

        socket.binaryType = 'arraybuffer'

        // Identify game socket by URL (always capture — handles reconnection)
        if (url.includes('socket.svr.colonist.io')) {
            const isNew = (gameSocket !== socket)
            gameSocket = socket
            gameSocketConfirmed = true
            if (isNew) {
                // Reset protocol state for fresh connection
                capturedServerId = null
                outgoingSequence = 0
                pendingSends = []
                console.log('[AICatan] Game socket captured:', url)
            }

            // Reset state when game socket closes (enables reconnection)
            socket.addEventListener('close', function (event) {
                console.log('[AICatan] Game socket closed (code=' + event.code + '). Ready for reconnection.')
                if (gameSocket === socket) {
                    gameSocket = null
                    gameSocketConfirmed = false
                }
            })
        }

        socket.addEventListener('message', function (event) {
            try {
                const msg = event.data
                let decoded

                if (msg instanceof ArrayBuffer) {
                    const raw = msgpack.decode(new Uint8Array(msg))
                    // Server→client msgpack wraps in {id, data} envelope; extract inner data
                    decoded = (raw && typeof raw === 'object' && 'data' in raw) ? raw.data : raw
                } else if (typeof msg === 'string') {
                    decoded = JSON.parse(msg)
                } else {
                    return
                }

                // Capture serverId from type 1 game setup message
                if (capturedServerId === null && decoded && decoded.type === 1
                    && decoded.payload && decoded.payload.serverId) {
                    capturedServerId = decoded.payload.serverId
                    console.log('[AICatan] CAPTURED serverId from game setup:', capturedServerId)
                    flushPendingSends()
                }

                // Only forward messages from the game socket
                if (gameSocketConfirmed && socket !== gameSocket) {
                    return
                }

                // Capture our player color from type 4 game init and cache it
                if (decoded && decoded.type === 4 && decoded.payload && decoded.payload.playerColor != null) {
                    myPlayerColor = decoded.payload.playerColor
                    console.log('[AICatan] Player color captured:', myPlayerColor)
                }

                const jsonStr = typeof decoded === 'object' ? JSON.stringify(decoded) : String(decoded)

                // Cache type 4 messages for resend on bot reconnect
                if (decoded && decoded.type === 4 && decoded.payload && decoded.payload.gameState) {
                    lastType4Message = jsonStr
                    console.log('[AICatan] Cached type 4 game init for reconnect')
                }

                sendToBot(jsonStr)
            } catch (e) {
                console.warn('[AICatan] Failed to decode message:', e.message)
            }
        })

        console.log('[AICatan] WebSocket intercepted:', url.substring(0, 60), gameSocketConfirmed ? '(GAME)' : '(other)')
        return socket
    }

    window.WebSocket.prototype = window._NativeWebSocket.prototype
    window.WebSocket.CONNECTING = window._NativeWebSocket.CONNECTING
    window.WebSocket.OPEN = window._NativeWebSocket.OPEN
    window.WebSocket.CLOSING = window._NativeWebSocket.CLOSING
    window.WebSocket.CLOSED = window._NativeWebSocket.CLOSED
}

// --- Initialization ---
function initialize() {
    if (typeof msgpack === 'undefined') {
        console.error('[AICatan] msgpack-lite not loaded! Check @require directive.')
        return
    }
    patchWebSocket()
    connectBotSocket()
    console.log('[AICatan] Userscript v3.0.0 initialized')
}

initialize()
