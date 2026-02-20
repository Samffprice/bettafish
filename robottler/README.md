# TODO
### General
 - [ ] Make sure userscript runs before other javascript to make sure WebSocket is intercepted (right now it seems like a 50/50 sometimes)
 - [ ] Completely decouple bot logic from colonist.io
 - [ ] Rewrite functions in naive_bot.py to work with arbitrary board states

### Bug fixes
 - [x] Continue turn after making trade with bank
 - [x] Keep track of turn information better

### Strategy
 - [x] Place initial settlements based on a score, which takes into account: current production, new production, harbor
   - [ ] Take expansion settlements into account in the score
 - [x] Improve calculate_next_purchase : if able to buy city buy city (after bank trades) -> if able to buy settlement buy settlement (after bank trades), etc
 - [ ] Think multiple steps ahead in trade (road -> settlement)
 - [x] Take into account possible bank trades when discarding cards

### Trade
 - [ ] Create 1:1 trades with opponent, 2:1 if it is a resource with < 3 own production

### Development cards
 - [x] Play knight cards when possible
 - [x] Implement all other playable development cards
