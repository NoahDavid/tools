import numpy as np

def getMinMaxSMA(data, sma_period=5, sma_shift=-1, fudge_acc=2, target_feature='open'):
    '''
    Get smoothed local minimums and maximums from data using SMA.

    ##### Parameters:
    data: The input data.

    sma_period: How much to smooth

    sma_shift: Correction factor: Moves the sma on top of the actual peaks.

    target_feature: Which feature to find maximums and minimums on.

    ##### Returns:

    A numpy array with the locations of each minimum (-1) and maximum (1), data values at those locations, and specs.
    '''
    
    feature_names = {
        'open_time': 0,
        'open': 1,
        'high': 2,
        'low': 3,
        'close': 4,
        'volume': 5,
        'close_time': 6,
        'quote_asset_volume': 7,
        'number_of_trades': 8,
        'taker_buy_base_asset_volume': 9,
        'taker_buy_quote_asset_volume': 10,
        'ignore': 11,
    }

    # Get index of feature:
    if target_feature in feature_names.keys():
        sma_index  = feature_names[target_feature]
    else:
        return np.array([])

    sma = [ 0 for i in range(len(data)) ]


    ind_shift = -2

    ind    = [ 0 for i in range(len(data)) ]
    prices = [ 0 for i in range(len(data)) ]


    for i in range(sma_period, len(data)):
        # Create the SMA
        buf = data[i - sma_period:i, sma_index]

        sma[i + sma_shift] = np.mean(buf)

    for i in range(2, len(ind)):
        # Find minimums and maximums

        inTop = (sma[i]     - sma[i - 1]) < 0
        inBot = (sma[i - 1] - sma[i - 2]) < 0

        isTurn = not inBot if inTop else inBot

        isBot  = isTurn and inBot
        isTop  = isTurn and inTop

        max = np.max([ data[i + ind_shift, 1], data[i + ind_shift, 4] ])
        min = np.min([ data[i + ind_shift, 1], data[i + ind_shift, 4] ])

        if isTop:
            # Put in a maximum:
            ind[i + ind_shift]    = 1
            prices[i + ind_shift] = max
        elif isBot:
            ind[i + ind_shift]    = -1
            prices[i + ind_shift] = min

    takeprofit_metrics = dict()

    for take_profit in np.arange(0.15, 1.5, 0.05):
        buy_locs = [ 0 for i in range(len(ind)) ]

        last_dwn    = -1
        # take_profit = 0.4
        stop_loss   = 0
        fees        = 0.15   # Per buy AND sell


        tp = 1 + take_profit / 100
        sl = 1 - stop_loss / 100
        fee = 1 - fees / 100

        state = 0

        for i in range(len(ind)):
            if ind[i] == -1:
                # Hit a low
                if (last_dwn >= 0 and prices[i] / prices[last_dwn] <= sl) or last_dwn < 0:
                    last_dwn = i
                
            if ind[i] == 1:
                # Hit a high

                if last_dwn >= 0:
                    if prices[i] / prices[last_dwn] >= tp:
                        # Set the bit
                        buy_locs[last_dwn] = 1
                        if last_dwn - 1 >= 0:
                            buy_locs[last_dwn - 1] = 1
                        if last_dwn + 1 < len(ind):
                            buy_locs[last_dwn + 1] = 1
                        last_dwn = -1

        data_den = np.sum(buy_locs) / len(buy_locs)
        accuracy_adj = np.sin(np.pi * data_den) * fudge_acc
        percent_profit = (tp * fee) ** (accuracy_adj * np.sum(buy_locs) / 3)
        days = len(buy_locs) / 60 / 24
        prft_day = percent_profit ** (1 / days)
        
        takeprofit_metrics[take_profit] = {
            'tp':             take_profit,
            'data_den':       data_den,
            'accuracy_adj':   accuracy_adj,
            'percent_profit': percent_profit,
            'prft_day':       prft_day
        }

    max_prft = 0

    for tp in takeprofit_metrics:
        tp = takeprofit_metrics[tp]
        if tp['prft_day'] > max_prft:
            max_prft = tp['prft_day']
            max_tp   = tp['tp']

    
    # data_den       = takeprofit_metrics[max_tp]['data_den']
    # accuracy_adj   = takeprofit_metrics[max_tp]['accuracy_adj']
    # percent_profit = takeprofit_metrics[max_tp]['percent_profit']
    # prft_day       = takeprofit_metrics[max_tp]['prft_day']

    # print(f'TP in question: {max_tp}\n\n')

    # print(f'Percentage of buys: {100 * data_den} %\n')
    # print(f'Accuracy Adjustment: {100 * accuracy_adj} %\n')

    # print(f'Total profit over {np.round(days)} days: {(percent_profit - 1) * 100} %\n')
    # print(f'Profit per day: {100 * (prft_day - 1)} %')

    return ind, prices, takeprofit_metrics[max_tp]

def getBuyLocs(max_mins, values, max_swing):
    '''
    Get the locations to buy.
    '''

    ind = max_mins
    prices = values
    max_tp = max_swing

    buy_locs = [ 0 for i in range(len(ind)) ]

    last_dwn    = -1
    take_profit = max_tp
    stop_loss   = 0
    fees        = 0.15   # Per buy AND sell


    tp = 1 + take_profit / 100
    sl = 1 - stop_loss / 100
    fee = 1 - fees / 100

    state = 0

    for i in range(len(ind)):
        if ind[i] == -1:
            # Hit a low
            if (last_dwn >= 0 and prices[i] / prices[last_dwn] <= sl) or last_dwn < 0:
                last_dwn = i
            
        if ind[i] == 1:
            # Hit a high

            if last_dwn >= 0:
                if prices[i] / prices[last_dwn] >= tp:
                    # Set the bit
                    buy_locs[last_dwn] = 1
                    if last_dwn - 1 >= 0:
                        buy_locs[last_dwn - 1] = 1
                    if last_dwn + 1 < len(ind):
                        buy_locs[last_dwn + 1] = 1
                    last_dwn = -1
    
    return buy_locs

def getLabelsQuick(data, short=True, thresh=0.4):

    locs = [ 0 for i in range(max(data.shape)) ]
    thres = 1 + thresh / 100

    for i in range(1, max(data.shape)):
        delta = data[i - 1, 2] / data[i, 3] if short else data[i, 2] / data[i - 1, 3]
        if delta >= thres:
            locs[i - 1] = 1
    
    return locs