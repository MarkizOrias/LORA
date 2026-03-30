"""
Trading LoRA Dataset Builder
==============================
Generates fine-tuning Q&A pairs in Phi-3 chat format from the ChromaDB
knowledge base. Combines:
  1. Hand-authored seed examples (high signal, covers every domain)
  2. Auto-generated examples from retrieved KB chunks (scale)

Output: trading_qa.jsonl  (Phi-3 ChatML format, ready for SFTTrainer)

Usage:
    python build_dataset.py --db ./trading-kb --output ./trading_qa.jsonl
"""

import json
import random
import argparse
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"

# ── Phi-3 chat format helper ──────────────────────────────────────────────────

def phi3_format(user_msg: str, assistant_msg: str) -> dict:
    """
    Returns a dict in Phi-3's ChatML format used by Unsloth's SFTTrainer.
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert trading analyst with deep knowledge of "
                    "market microstructure, macro economics, technical analysis, "
                    "quantitative methods, cryptocurrency on-chain dynamics, "
                    "commodities, and professional risk management. "
                    "You reason like a senior hedge fund portfolio manager. "
                    "Be precise, cite relevant market mechanics, and always "
                    "address risk alongside opportunity. Never give financial "
                    "advice — give analytical frameworks and factual reasoning."
                )
            },
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


# ── Seed examples — hand-authored, one per knowledge domain ──────────────────
# These are the highest-quality examples. Every domain must be represented.
# Quantity matters less than quality here — 10 brilliant examples beat 100 weak.

SEED_EXAMPLES = [

    # ── MICROSTRUCTURE ────────────────────────────────────────────────────────
    phi3_format(
        "BTC just printed a 3% green candle on the 1h chart with volume 4x the "
        "20-period average. Is this a reliable breakout signal?",
        "Volume confirmation is necessary but not sufficient. A 4x volume spike "
        "on an up-candle is meaningful — it indicates genuine participation rather "
        "than a low-liquidity gap fill. However, before calling it a reliable "
        "breakout I would check three things.\n\n"
        "First, what is the order flow composition? On a centralised exchange, "
        "look at the taker buy/sell ratio. If the volume is predominantly taker "
        "buying (aggressive market orders lifting the ask), that is informed "
        "directional flow. If it is maker-driven, it may be a large limit order "
        "absorbed by both sides — less directionally meaningful.\n\n"
        "Second, where is price relative to the bid-ask spread and the order book "
        "depth? A breakout through a thin book is cheap to manufacture and often "
        "reverses as liquidity refills. A breakout through a thick book — where "
        "the spike consumed significant offer depth — is structurally stronger.\n\n"
        "Third, check exchange net flows. If long-term holders are simultaneously "
        "moving BTC onto exchanges, the volume spike may be absorption of "
        "distribution rather than accumulation. Divergence between price action "
        "and on-chain net flows is a classic warning sign.\n\n"
        "Risk framing: if you enter on this signal, place a stop below the base "
        "of the candle. A genuine breakout should not revisit the breakout level "
        "— if it does within two bars, the probability of a false breakout rises "
        "sharply and you exit at 1R loss."
    ),

    # ── MACRO / RATES ─────────────────────────────────────────────────────────
    phi3_format(
        "The Fed just surprised markets with a 50bp rate cut when only 25bp was "
        "priced. Walk me through the expected cross-asset impact in the next "
        "24-48 hours.",
        "A 25bp surprise cut is a significant dovish shock. The transmission "
        "mechanism works as follows.\n\n"
        "Equities: risk assets typically rally on the announcement as the "
        "discount rate for future earnings falls. However, differentiate between "
        "a 'good cut' (insurance cut in a healthy economy) and a 'bad cut' "
        "(emergency cut signalling recession fear). A 50bp surprise with no "
        "concurrent recession language is equity-positive. Watch whether the S&P "
        "holds the initial spike — if it fades within two hours, the market is "
        "reading the cut as distress, not stimulus.\n\n"
        "Fixed income: the front end of the yield curve (2-year) drops sharply "
        "and immediately. The long end (10-year, 30-year) is more complex — a "
        "growth-positive cut can actually steepen the curve as inflation "
        "expectations rise. Watch the 2s10s spread direction; steepening is "
        "a risk-on signal.\n\n"
        "Dollar: a dovish surprise typically weakens the DXY as rate "
        "differentials compress. This is positive for commodities priced in USD "
        "(gold, oil, BTC) in the near term.\n\n"
        "BTC specifically: crypto tends to lag equities by 30-90 minutes on macro "
        "events as the market is dominated by retail participants who process news "
        "more slowly. The initial move in BTC is often weaker than equities; the "
        "follow-through over 24-48 hours is typically stronger if equities confirm "
        "the bullish read.\n\n"
        "Risk: the surprise element means short-term volatility is elevated in "
        "both directions. Size positions at 50-75% of normal during the first "
        "two hours. Let the initial whipsaw settle."
    ),

    # ── CRYPTO ON-CHAIN ───────────────────────────────────────────────────────
    phi3_format(
        "BTC's MVRV ratio is at 3.2 and SOPR for long-term holders just crossed "
        "above 1.5. What does this tell me about where we are in the cycle?",
        "These two signals together paint a picture of late-stage bull market "
        "distribution territory, but not necessarily an immediate top.\n\n"
        "MVRV (Market Value to Realised Value) at 3.2 means the average coin is "
        "sitting at 220% unrealised profit relative to its acquisition cost. "
        "Historically, MVRV above 3.0 corresponds to the 'euphoria zone' in "
        "Bitcoin cycles — the 2017 cycle peaked around MVRV 4.7, the 2021 cycle "
        "peaked near 3.9 in April and 3.6 in November. You are in the zone where "
        "long-term holder profit-taking accelerates, but the cycle can extend "
        "further.\n\n"
        "LTH-SOPR above 1.5 means long-term holders (coins unmoved for 155+ days) "
        "are on average selling at 50% profit. This confirms active distribution "
        "by the cohort with the strongest conviction — they accumulated at lower "
        "prices and are now realising gains. When LTH-SOPR is elevated and "
        "trending upward, watch whether exchange inflows from LTH wallets are "
        "increasing concurrently — if yes, the distribution is organised.\n\n"
        "Tactical read: this environment calls for asymmetric positioning. You can "
        "still be long, but with tighter stops and smaller size than earlier in "
        "the cycle. Consider scaling out 20-30% of exposure at current levels. "
        "The signal to exit the bulk of long exposure is MVRV crossing back below "
        "2.4 from above combined with LTH-SOPR declining — that sequence has "
        "historically marked the transition from distribution to capitulation."
    ),

    # ── RISK / POSITION SIZING ────────────────────────────────────────────────
    phi3_format(
        "I have a trading signal with a historical win rate of 55% and an average "
        "win-to-loss ratio of 1.8:1. How should I size my positions?",
        "With a 55% win rate and 1.8:1 reward-to-risk ratio, your edge is "
        "positive and quantifiable. Let me work through the sizing framework.\n\n"
        "Expectancy per trade: (0.55 × 1.8R) - (0.45 × 1R) = 0.99R - 0.45R "
        "= 0.54R. For every 1R risked, you expect to gain 0.54R on average. "
        "This is a solid edge — better than many professional systems.\n\n"
        "Full Kelly fraction: f* = (bp - q) / b = (1.8 × 0.55 - 0.45) / 1.8 "
        "= (0.99 - 0.45) / 1.8 = 0.30. The full Kelly suggests risking 30% of "
        "capital per trade. In practice this is far too aggressive — Kelly "
        "assumes perfect knowledge of your edge, and real-world edges fluctuate "
        "with regime changes.\n\n"
        "Practical sizing: use half-Kelly or quarter-Kelly. Half-Kelly here gives "
        "15% risk per trade, which is still too high for most portfolios. I would "
        "recommend quarter-Kelly: 7-8% of capital at risk per trade.\n\n"
        "However, risk per trade is not the same as position size. If your stop "
        "is 2% below entry on a stock, a 7% capital-at-risk means a position size "
        "of 350% of capital — using leverage. If you are not using leverage, "
        "calibrate: risk per trade = (position size as % of capital) × (stop "
        "distance as % of entry). For a 2% stop and a 1% capital risk limit, "
        "position size is 50% of portfolio.\n\n"
        "A practical starting point: cap risk per trade at 1-2% of capital until "
        "you have 50+ live trades confirming the backtest edge holds out-of-sample. "
        "The edge you measured is on historical data — live market slippage and "
        "regime changes will compress it."
    ),

    # ── COMMODITIES ───────────────────────────────────────────────────────────
    phi3_format(
        "Crude oil futures are in contango with the front month at $72 and the "
        "12-month contract at $78. What are the trading implications?",
        "A contango structure with a $6 spread over 12 months (roughly 8.3% "
        "annualised) tells you several things about market structure and "
        "constrains your trading approach.\n\n"
        "What contango signals: The market is pricing in either (a) ample current "
        "supply relative to demand, (b) high storage availability and low storage "
        "cost, or (c) expectation of higher future prices due to supply constraints "
        "ahead. In crude, steep contango typically reflects near-term "
        "oversupply — refineries are not scrambling for prompt barrels.\n\n"
        "Impact on long positions: If you buy the front-month contract and roll "
        "it forward each month, you pay the contango cost on every roll. At 8.3% "
        "annualised, a buy-and-hold long position in crude futures loses "
        "approximately 0.7% per month to roll decay even if spot prices are flat. "
        "This is why commodity ETFs that roll futures (like USO historically) "
        "significantly underperform spot crude during contango markets.\n\n"
        "Trading opportunities: (1) Calendar spread — short the front, long the "
        "12-month. You earn the carry if contango narrows (e.g., if a supply "
        "disruption tightens the prompt market). (2) Wait for the term structure "
        "to shift to backwardation before taking outright longs — backwardation "
        "means the market is rewarding you for holding front-month exposure.\n\n"
        "Watch the EIA weekly inventory report — a sustained draw in Cushing "
        "storage is the trigger that historically collapses contango toward flat "
        "or backwardation."
    ),

    # ── TECHNICAL ANALYSIS / REGIME ───────────────────────────────────────────
    phi3_format(
        "The 50-day MA just crossed below the 200-day MA on the daily chart for "
        "ETH. How significant is this death cross and how do I trade it?",
        "The death cross is one of the most widely discussed signals in technical "
        "analysis and also one of the most frequently misused. The honest answer "
        "is: it is a lagging confirmation of a trend change, not a predictive "
        "signal.\n\n"
        "Why it lags: moving averages are averages of past prices. By the time "
        "the 50-day crosses the 200-day, price has usually already declined 15-30% "
        "from its peak in crypto markets. The signal tells you what already "
        "happened, not what will happen next. In 2022, ETH's death cross occurred "
        "in January — after ETH had already fallen from $4,800 to $2,400. Shorting "
        "on the cross alone captured only a portion of the eventual move and "
        "exposed you to violent counter-trend rallies.\n\n"
        "When it is useful: the death cross is most reliable as a regime "
        "confirmation tool rather than an entry signal. Once the cross occurs, it "
        "shifts your prior — trend-following signals should be taken more "
        "aggressively, mean-reversion long signals should be taken with smaller "
        "size or skipped entirely.\n\n"
        "Practical approach: rather than shorting on the cross itself, wait for "
        "a relief rally back toward the 200-day MA (now resistance) and short "
        "that test with a stop above the 200-day. This gives a better "
        "risk/reward entry. Alternatively, use the cross as a filter: only take "
        "short signals from your primary system, suppress long signals, until "
        "either a golden cross occurs or price reclaims the 200-day convincingly "
        "on volume."
    ),

    # ── ML / SYSTEMATIC ───────────────────────────────────────────────────────
    phi3_format(
        "My LightGBM model shows a Sharpe ratio of 2.1 in backtesting but only "
        "0.4 in live trading over the first three months. What is likely wrong?",
        "A gap this large between backtesting and live performance is a classic "
        "pattern and the causes are usually identifiable.\n\n"
        "Most likely cause — lookahead bias: if any feature in your model was "
        "computed using data not available at the time of the signal, your backtest "
        "is inflated. Common sources: using the close price of the current bar to "
        "generate a signal that executes at that same close; using a volume "
        "indicator calculated on the full day's volume before the day closes; "
        "or survivorship bias in your training universe (only backtesting on "
        "stocks or coins that still exist, not ones that went to zero).\n\n"
        "Second cause — overfitting: a Sharpe of 2.1 on financial data without "
        "walk-forward validation is suspicious. Financial time series have low "
        "signal-to-noise ratios. If you trained on 3 years and validated on a "
        "random holdout, you likely found patterns specific to the training period. "
        "Test: take your model's signals from the last 6 months of your training "
        "period (before any holdout) — if Sharpe drops significantly there, "
        "overfitting is the cause.\n\n"
        "Third cause — execution assumptions: backtests typically assume filling "
        "at the signal bar's close. Live trading fills at the next open or worse, "
        "with slippage. For liquid instruments this may be 0.05-0.2%; for "
        "illiquid crypto pairs or small-cap stocks it can be 0.5-2% per trade "
        "— easily destroying a moderate edge.\n\n"
        "Diagnosis path: (1) audit every feature for lookahead; (2) re-run with "
        "combinatorial purged cross-validation; (3) add realistic slippage of "
        "0.1% per trade to the backtest and observe Sharpe degradation; "
        "(4) compare live signal distribution to backtest signal distribution — "
        "if the model is generating different signal frequencies or magnitudes "
        "live, regime shift is the cause."
    ),

    # ── SENTIMENT / BEHAVIOURAL ───────────────────────────────────────────────
    phi3_format(
        "Bitcoin's funding rate on perpetual futures has been consistently "
        "positive at 0.05-0.08% per 8 hours for the past two weeks. What does "
        "this mean for positioning?",
        "A sustained funding rate of 0.05-0.08% per 8 hours annualises to "
        "roughly 55-87% — extremely high. This is telling you the derivatives "
        "market is heavily long-biased and longs are paying shorts to maintain "
        "their positions.\n\n"
        "What this signals: extreme crowding on the long side. Perpetual futures "
        "funding acts as a sentiment thermometer — when everyone wants to be long, "
        "they push the contract above spot, and the funding mechanism (longs pay "
        "shorts) restores equilibrium. Sustained high positive funding means the "
        "crowd's conviction has overridden the cost.\n\n"
        "The liquidation cascade risk: these long positions have liquidation levels "
        "below current price. If price drops sharply — on any negative catalyst or "
        "even just a large spot sale — a cascade of forced liquidations can "
        "amplify the move dramatically. This is mechanically why high-funding "
        "regimes often precede violent corrections: not because funding is the "
        "cause, but because it is evidence of the crowded positioning that makes "
        "cascades possible.\n\n"
        "Tactical read: in a high-funding environment, avoid adding to longs. "
        "If you are already long, consider hedging with a small short position "
        "on perps — you collect the funding rate (shorts receive, longs pay) "
        "while maintaining spot exposure. The classic trade is to be long spot "
        "and short an equivalent amount of perps: delta-neutral but collecting "
        "8-15% annualised yield from funding. Exit the hedge when funding "
        "normalises below 0.01% per 8 hours."
    ),

    # ── VALUATION / EQUITIES ──────────────────────────────────────────────────
    phi3_format(
        "A gold mining stock I'm watching is trading at 12x EV/EBITDA while the "
        "gold price is at $2,400. The stock was at 18x EV/EBITDA when gold was "
        "at $1,900. Is it cheap?",
        "Valuing a gold miner requires understanding the leverage structure — "
        "miners are not just a proxy for gold price; they are a leveraged "
        "derivative of the gold price with fixed-cost operating structures.\n\n"
        "Why the multiple compressed: the 33% compression from 18x to 12x despite "
        "a 26% rise in gold price is counterintuitive but common in mining stocks. "
        "Possible explanations: (1) operating cost inflation — if the miner's "
        "all-in sustaining cost (AISC) rose from $1,200 to $1,600 per ounce, "
        "the margin expansion from higher gold is smaller than expected; "
        "(2) capital allocation concerns — miners that reinvest into expensive "
        "acquisitions or dilutive equity raises at peak cycle are penalised; "
        "(3) production disappointments — if reserves or production guidance was "
        "cut, the market is discounting future output.\n\n"
        "The correct valuation anchor: instead of a trailing EV/EBITDA multiple, "
        "build a simple NAV model. Take proven reserves in ounces, subtract AISC "
        "per ounce from the current gold price, apply a discount rate (typically "
        "8-12% for senior miners), and compare to market cap. If NAV at $2,400 "
        "gold implies significant upside to current price, the stock is cheap on "
        "fundamentals.\n\n"
        "Key risk: gold miners have operating leverage that works both ways. A "
        "10% decline in gold price with flat costs can wipe out 30-50% of EBITDA "
        "depending on the cost structure. Size positions in miners smaller than "
        "equivalent gold exposure — typically 50-70% of the notional you would "
        "allocate to physical gold."
    ),

    # ── REGIME DETECTION ─────────────────────────────────────────────────────
    phi3_format(
        "How do I know whether I'm in a trending market or a mean-reverting "
        "market for a given instrument, and why does it matter for my strategy?",
        "Regime detection is arguably more important than signal generation — "
        "applying a trend-following strategy in a choppy mean-reverting market, "
        "or a mean-reversion strategy in a strong trend, are both reliably "
        "loss-generating.\n\n"
        "Quantitative detection methods:\n\n"
        "1. Hurst exponent: values above 0.5 indicate persistence (trending), "
        "below 0.5 indicate anti-persistence (mean-reverting), at 0.5 is random "
        "walk. Calculate over a rolling 100-200 bar window. For daily data on "
        "liquid markets, the Hurst exponent oscillates — use it as a probability "
        "weight rather than a binary switch.\n\n"
        "2. ADX (Average Directional Index): above 25 indicates a trending regime, "
        "below 20 indicates ranging. ADX above 40 is a strong trend, and "
        "paradoxically above 50 often signals trend exhaustion as the move becomes "
        "parabolic.\n\n"
        "3. Variance ratio test: compare the variance of k-period returns to k "
        "times the variance of 1-period returns. A ratio significantly above 1 "
        "indicates momentum; significantly below 1 indicates mean reversion.\n\n"
        "Why it matters for strategy selection: a trend-following system (moving "
        "average crossovers, breakout entries) has positive expected value in "
        "trending regimes and systematically negative expected value in "
        "mean-reverting regimes — it buys breakouts that reverse immediately. "
        "Conversely, a mean-reversion system (Bollinger Band fades, RSI extremes) "
        "thrives in choppy markets but is catastrophically wrong in strong trends "
        "— you keep fading rallies that continue higher.\n\n"
        "Practical implementation: maintain two strategy modules. Use regime "
        "detection as a gating filter — activate the trend module when ADX > 25 "
        "and Hurst > 0.55; activate mean-reversion when ADX < 20 and Hurst < 0.48; "
        "reduce position size or go flat in the ambiguous middle zone."
    ),

    # ── INTERMARKET / CROSS-ASSET ─────────────────────────────────────────────
    phi3_format(
        "Gold is making new all-time highs while both equities and BTC are also "
        "near all-time highs. Is this historically unusual and what does it signal?",
        "Yes, this is historically unusual and warrants close attention. Gold, "
        "equities, and BTC moving to all-time highs simultaneously represents a "
        "conflicting signal set because gold and equities traditionally have "
        "negative to zero correlation in risk-on/risk-off frameworks.\n\n"
        "Historical context: gold typically appreciates during risk-off periods "
        "(equity selloffs, recessions, geopolitical stress) because it is a safe "
        "haven and a dollar hedge. When gold makes ATH alongside equities, it "
        "suggests that the driver of gold's move is not risk aversion but "
        "currency debasement fear — specifically, concerns about fiscal deficits, "
        "dollar reserve status erosion, or anticipated inflation that makes real "
        "rates negative.\n\n"
        "What this environment has historically preceded: the 2020 analog is "
        "instructive — gold surged alongside equities as QE expanded balance "
        "sheets globally. Both assets rose together until the dollar "
        "strengthened on rate hike expectations, at which point gold corrected "
        "while equities continued.\n\n"
        "BTC's role in this dynamic: if BTC is also at ATH, the joint move "
        "suggests all three are benefiting from the same macro theme — excess "
        "liquidity seeking stores of value and growth assets simultaneously. "
        "This regime typically ends with a dollar spike (DXY breaking higher) "
        "which hits gold first, then BTC, then growth equities in sequence.\n\n"
        "Monitor: DXY trend, real 10-year yields (TIPS), and central bank "
        "forward guidance. A shift toward hawkishness is the regime-ending "
        "catalyst. Until that signal, the trend in all three assets remains "
        "intact — but size down and tighten stops."
    ),
]


# ── Auto-generation from KB chunks ───────────────────────────────────────────

AUTO_GEN_QUERIES = [
    # Macro
    ("macro", "Explain how a yield curve inversion signals a recession and the typical lag time"),
    ("macro", "How does quantitative tightening affect liquidity in risk assets"),
    ("macro", "What happens to commodity prices when the dollar index rises sharply"),
    ("macro", "How do central bank balance sheet expansions affect asset prices"),
    # Microstructure
    ("microstructure", "What is adverse selection risk and how does it affect market makers"),
    ("microstructure", "How does order flow imbalance predict short-term price direction"),
    ("microstructure", "Explain the role of dark pools in modern equity markets"),
    ("microstructure", "What is market impact and how does it affect large order execution"),
    # Crypto
    ("crypto", "How does the Bitcoin halving affect miner revenue and selling pressure"),
    ("crypto", "What does exchange reserve decline indicate about Bitcoin market structure"),
    ("crypto", "Explain the relationship between ETH staking yield and ETH price"),
    ("crypto", "What is the Puell Multiple and when does it signal miner capitulation"),
    # Risk
    ("risk", "When should a trader reduce position size during a drawdown"),
    ("risk", "How does correlation between positions affect portfolio risk"),
    ("risk", "Explain the difference between volatility-based and R-multiple position sizing"),
    # Technical
    ("technical", "How does volume profile analysis identify key support and resistance levels"),
    ("technical", "When is the RSI overbought reading a reliable sell signal versus a trend signal"),
    ("technical", "What makes a candlestick pattern more or less reliable across timeframes"),
    # Systems / ML
    ("systems", "What is the difference between in-sample and out-of-sample testing in trading"),
    ("systems", "How does the triple barrier method improve trade labelling for ML models"),
    ("systems", "Why do most backtested strategies fail in live trading"),
    # Commodities
    ("technical", "How does seasonality affect crude oil prices across the calendar year"),
    ("macro", "What is the Commitment of Traders report and how do traders use it"),
]


def generate_auto_examples(collection, embedder, n_per_query: int = 2) -> list[dict]:
    """
    For each query, retrieve the top-k chunks from the KB and format them
    as a question + grounded answer pair.
    The 'answer' is constructed from the retrieved chunk text to ensure
    it is knowledge-base-grounded, not hallucinated.
    """
    examples = []

    for domain, query in AUTO_GEN_QUERIES:
        vec = embedder.encode([query])[0].tolist()
        results = collection.query(
            query_embeddings=[vec],
            n_results=3,
            where={"domain": domain} if domain else None,
            include=["documents", "metadatas"],
        )

        if not results["documents"] or not results["documents"][0]:
            continue

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        # Build a grounded answer by combining top chunks
        context = "\n\n".join(docs[:2])
        source  = metas[0].get("source", "knowledge base")

        # Format as a study-style Q&A
        example = phi3_format(
            user_msg=query + "?",
            assistant_msg=(
                f"Based on {source}:\n\n{context}\n\n"
                "Apply this to your trading by considering both the "
                "opportunity and the risk dimension of the above framework."
            )
        )
        examples.append(example)

    return examples


# ── Main ──────────────────────────────────────────────────────────────────────

def build_dataset(db_path: Path, output_path: Path, n_auto: int = 100):
    print(f"\nLoading KB from {db_path} ...")
    client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection("trading_static")
    print(f"KB chunks available: {col.count():,}")

    print("Loading embedding model ...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"Seed examples: {len(SEED_EXAMPLES)}")

    print("Generating auto examples from KB ...")
    auto_examples = generate_auto_examples(col, embedder)
    print(f"Auto examples generated: {len(auto_examples)}")

    all_examples = SEED_EXAMPLES + auto_examples

    # Shuffle to avoid order bias in training
    random.seed(42)
    random.shuffle(all_examples)

    # Write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nDataset written: {output_path}")
    print(f"Total examples : {len(all_examples)}")
    print(f"\nBreakdown:")
    print(f"  Seed (hand-authored) : {len(SEED_EXAMPLES):>4}")
    print(f"  Auto (KB-grounded)   : {len(auto_examples):>4}")
    print(f"\nNext step: run finetune_phi3.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",     type=Path, default=Path("./trading-kb"))
    parser.add_argument("--output", type=Path, default=Path("./trading_qa.jsonl"))
    args = parser.parse_args()
    build_dataset(args.db, args.output)
