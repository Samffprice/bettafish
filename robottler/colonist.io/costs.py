from purchase_types import PurchaseType
from resources import Resources

COSTS = {PurchaseType.ROAD:       {Resources.WOOD: 1, Resources.BRICK: 1},
         PurchaseType.SETTLEMENT: {Resources.WOOD: 1, Resources.BRICK: 1,
                                   Resources.SHEEP: 1, Resources.WHEAT: 1},
         PurchaseType.CITY:       {Resources.WHEAT: 2, Resources.ORE: 3},
         PurchaseType.DEV_CARD:   {Resources.SHEEP: 1, Resources.WHEAT: 1, Resources.ORE: 1}}
