import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def channel_locations(Th, Rd, channels):

    # ?------------------------------------------- Check type data -------------------------------
    if 'Series' not in str(type(Th)):      # Check type Th
        Th = pd.Series(Th)
    if 'Series' not in str(type(Rd)):      # Check type Rd
        Rd = pd.Series(Rd)
    # *-------------------------------------- Convert degree to radian ---------------------------
    if Th.max() > 20:
        Th = (np.pi/180)*Th
    elif Rd.max() > 20:
        Rd = (np.pi/180)*Rd
    sq = 0.5/max(min(1.0, max(Rd)*1.02), 0.5)
    # Rd = Rd * sq
    # ---------------------------------------------- Circle --------------------------------------
    x = Rd * np.cos(Th) * sq
    y = Rd * np.sin(Th) * sq
    circ = np.linspace(start=0, stop=2*np.pi, num=200).flatten()
    rx = np.sin(circ).tolist()
    ry = np.cos(circ).tolist()
    hx = np.dot(rx, 1.06)
    hy = np.dot(ry, 1.06)
    # ---------------------------------------------- right & left ear -----------------------------
    EarX = np.dot(
        [.497-.005, .510, .518, .5299, .5419, .54, .547, .532, .510, .489-.005], 1.13)
    EarY = np.dot([0.04+.0555, 0.04+.0775, 0.04+.0783, 0.04+.0746,
                  0.02+.0555, -.0055, -.0932, -.1313, -.1384, -.1199], 1.13)
    # !------------------------------------------- Set figure -------------------------------------
    _, axs = plt.subplots(nrows=1, sharey='row',
                            figsize=(2.7, 2.7), facecolor='#D6D6D6')

    """hin = 1.06                 # hin = sq*plotrad*(1- 0.007/2)   # inner head ring radius
    hx = np.dot(rx,hin)
    hy = np.dot(ry,hin)
    axs.plot(np.dot([0.09, 0.02, 0, -0.02, -0.09],sq), np.dot([0.4954, 0.57, 0.575, 0.57, 0.4954], sq), 'k') # plot nosث
    """

    axs.plot(np.dot(hx, max(y)), np.dot(hy, max(y)), 'k')
    axs.plot(np.dot(EarX, sq), np.dot(EarY, sq), 'k')    # plot right ear
    axs.plot(-np.dot(EarX, sq), np.dot(EarY, sq), 'k')   # plot left ear
    axs.plot(np.dot([0.08, 0.01, 0, -0.01, -0.08], 0.775),
             np.dot([0.4954, 0.57, 0.575, 0.57, 0.4954], 0.775), 'k')  # plot nose

    axs.plot(y, x, '.')
    for ind, val in enumerate(channels):
        axs.text(y[ind], x[ind]+0.0255, val, fontsize=5.5,
                 horizontalalignment='center', verticalalignment='center')


    axs.set_axis_off()
    axs.set_title('Channel locations',  fontsize=8.5)
    axs.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)