// ==UserScript==
// @name             Colonist interceptor
// @include          /^https://colonist.io/
// @run-at           document-start
// @grant            none
// ==/UserScript==

const colonistioActions = Object.freeze({
    THROW_DICE: "15",
    MOVE_ROBBER: "16",
    ROB_PLAYER: "18",
    PASS_TURN: "19",
    SELECT_CARDS: "20",
    BUY_DEVELOPMENT_CARD: "21",
    WANT_BUILD_ROAD: "22",
    BUILD_ROAD: "23",
    WANT_BUILD_SETTLEMENT: "25",
    BUILD_SETTLEMENT: "26",
    WANT_BUILD_CITY: "27",
    BUILD_CITY: "28",
    PLAY_DEVELOPMENT_CARD: "52",
    CREATE_TRADE: "54",
    ACCEPT_TRADE: "55",
    REJECT_TRADE: "57"
})

const devCards = Object.freeze({
    ROBBER: 7,
    VICTORY_POINT: 8,
    MONOPOLY: 9,
    ROAD_BUILDING: 10,
    YEAR_OF_PLENTY: 11
})

const EncoderModule = (function () {
    let instance
    /* eslint-disable */
    function createInstance () {
        const e = window.webpackJsonp[0][1] // Modules used by colonist.io are hidden here
        const i = {}
        function o (t) {
            if (i[t]) { return i[t].exports }
            const a = i[t] = {
                i: t,
                l: !1,
                exports: {}
            }
            return e[t].call(a.exports, a, a.exports, o),
            a.l = !0,
            a.exports
        }
        o.m = e,
        o.c = i,
        o.d = function (e, t, a) {
            o.o(e, t) || Object.defineProperty(e, t, {
                enumerable: !0,
                get: a
            })
        }
        ,
        o.r = function (e) {
            typeof Symbol !== "undefined" && Symbol.toStringTag && Object.defineProperty(e, Symbol.toStringTag, {
                value: "Module"
            }),
            Object.defineProperty(e, "__esModule", {
                value: !0
            })
        }
        ,
        o.t = function (e, t) {
            if (1 & t && (e = o(e)),
            8 & t) { return e }
            if (4 & t && typeof e === "object" && e && e.__esModule) { return e }
            const a = Object.create(null)
            if (o.r(a),
            Object.defineProperty(a, "default", {
                enumerable: !0,
                value: e
            }),
            2 & t && typeof e !== "string") {
                for (const i in e) {
                    o.d(a, i, function (t) {
                        return e[t]
                    }
                        .bind(null, i))
                }
            }
            return a
        }
        ,
        o.n = function (e) {
            const t = e && e.__esModule
                ? function () {
                    return e.default
                }
                : function () {
                    return e
                }

            return o.d(t, "a", t),
            t
        }
        ,
        o.o = function (e, t) {
            return Object.prototype.hasOwnProperty.call(e, t)
        }
        ,
        o.p = "/dist"

        return o(548) // The module we need is located in window.webpackJsonp[0][1][540]
    }
    /* eslint-enable */

    return {
        getInstance: function () {
            if (!instance) {
                instance = createInstance()
            }
            return instance
        }
    }
})()

const botSocket = new WebSocket("ws://localhost:8765")
let gameSocket
const NativeWebSocket = window.WebSocket
window.WebSocket = function (...args) {
    const socket = new NativeWebSocket(...args)
    socket.addEventListener("message", function (event) {
        const msg = event.data
        let data = EncoderModule.getInstance().decode(msg).data
        if (typeof data === "object") data = JSON.stringify(data)
        botSocket.send(data)
    })
    const nativeSend = socket.send
    socket.send = function (data) {
        const decodedData = EncoderModule.getInstance().decode(data)
        console.log(decodedData)
        nativeSend.apply(this, [data])
    }
    gameSocket = socket
    return socket
}

botSocket.onmessage = function (event) {
    const parsedData = JSON.parse(event.data)
    const tradeData = {}

    switch (parsedData.action) {
    case 0: // Build road
        sendEncoded({ id: colonistioActions.WANT_BUILD_ROAD, data: true }) // data: road id
        sendEncoded({ id: colonistioActions.BUILD_ROAD, data: parsedData.data }) // data: road id
        break
    case 1: // Build settlement
        sendEncoded({ id: colonistioActions.WANT_BUILD_SETTLEMENT, data: true }) // data: road id
        sendEncoded({ id: colonistioActions.BUILD_SETTLEMENT, data: parsedData.data }) // data: settlement id
        break
    case 2: // Build city
        sendEncoded({ id: colonistioActions.WANT_BUILD_CITY, data: true }) // data: road id
        sendEncoded({ id: colonistioActions.BUILD_CITY, data: parsedData.data }) // data: settlement id
        break
    case 3: // Buy development card
        sendEncoded({ id: colonistioActions.BUY_DEVELOPMENT_CARD, data: true })
        break
    case 4: // Throw dice
        sendEncoded({ id: colonistioActions.THROW_DICE, data: true })
        break
    case 5: // Pass turn
        sendEncoded({ id: colonistioActions.PASS_TURN, data: true })
        break
    case 6: // Accept trade
        sendEncoded({ id: colonistioActions.ACCEPT_TRADE, data: parsedData.data })
        break
    case 7: // Reject trade
        sendEncoded({ id: colonistioActions.REJECT_TRADE, data: parsedData.data })
        break
    case 8: // Move robber
        sendEncoded({ id: colonistioActions.MOVE_ROBBER, data: parsedData.data })
        break
    case 9: // Rob player
        sendEncoded({ id: colonistioActions.ROB_PLAYER, data: parsedData.data })
        break
    case 10: // Discard cards
        sendEncoded({ id: colonistioActions.SELECT_CARDS, data: parsedData.data })
        break
    case 11: // Trade with bank
        // Todo fix the player indexes for other than standard bot games

        tradeData.actions = [{ player: 2, allowedTradeActions: [] }, { player: 3, allowedTradeActions: [] }, { player: 4, allowedTradeActions: [] }]
        tradeData.activeTarges = [2, 3, 4]
        tradeData.allowableTradeResources = [0]
        tradeData.creator = 1
        tradeData.id = "0"
        tradeData.isCounterOffer = false
        tradeData.offeredResources = { allowableCardTypes: [0], cards: parsedData.data.offered }
        tradeData.responses = [{ player: 2, response: 2 }, { player: 3, response: 2 }, { player: 4, response: 2 }]
        tradeData.targets = [2, 3, 4]
        tradeData.wantedResources = { allowableCardTypes: [0], cards: parsedData.data.wanted }

        sendEncoded({ id: colonistioActions.CREATE_TRADE, data: tradeData })
        break
    case 12: // Play development card
        sendEncoded({ id: colonistioActions.PLAY_DEVELOPMENT_CARD, data: parsedData.data })
        break
    }
}

function sendEncoded (data) {
    const encodedData = EncoderModule.getInstance().encode(data)
    gameSocket.send(encodedData)
}
