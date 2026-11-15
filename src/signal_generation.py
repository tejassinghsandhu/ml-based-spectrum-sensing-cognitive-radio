import numpy as np
def generate_bpsk(n_symbols):

    bits = np.random.randint(0, 2, n_symbols)

    symbols = 2 * bits - 1

    return symbols.astype(complex)


bpsk = generate_bpsk(1000)

print("BPSK generated:", len(bpsk))
def generate_qpsk(n_symbols):

    i = np.random.choice([-1, 1], n_symbols)
    q = np.random.choice([-1, 1], n_symbols)

    symbols = (i + 1j*q) / np.sqrt(2)

    return symbols


qpsk = generate_qpsk(1000)

print("QPSK generated:", len(qpsk))

def generate_16qam(n_symbols):

    levels = np.array([-3, -1, 1, 3])

    i = np.random.choice(levels, n_symbols)
    q = np.random.choice(levels, n_symbols)

    symbols = (i + 1j*q) / np.sqrt(10)

    return symbols


qam16 = generate_16qam(1000)

print("16-QAM generated:", len(qam16))