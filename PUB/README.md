# 🌪️ Cyclone: Public Key search only..

## 🚀 Usage

```bash
./Cyclone -k <public_key_hex> [-p <puzzle> | -r <startHex:endHex>] -b <prefix_length> [-t <threads>]
  -k : Target compressed public key (66 hex chars)
  -b : Number of prefix bytes to compare (1-33)
  -t : Number of CPU threads to use (default: all available cores
```

```bash
./Cyclone -k 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 -p 135 -t 12 -b 6
```

## ✌️**TIPS**
BTC: bc1qdwnxr7s08xwelpjy3cc52rrxg63xsmagv50fa8
