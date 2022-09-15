from scipy import signal


def filtering(data, f_low, f_high, order, fs, btype):
    f_low = f_low / (fs / 2)
    f_high = f_high / (fs / 2)
    if btype == "low":
        b, a = signal.butter(order, f_low, btype='low')
    elif btype == "high":
        b, a = signal.butter(order, f_high, btype='high')
    elif btype == "bandpass":
        b, a = signal.butter(order, [f_low, f_high], btype='bandpass')
    elif btype == "bandstop":
        b, a = signal.butter(order, [f_low, f_high], btype='bandstop')
    data_filter = signal.filtfilt(b, a, data)
    return data_filter
