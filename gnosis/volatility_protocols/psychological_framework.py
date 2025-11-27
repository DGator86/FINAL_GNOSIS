"""
Psychological Framework for Volatility Trading
===============================================

The 10 psychological demons that destroy vol traders and their permanent fixes.

Volatility trading is 20% math and 80% surviving your own brain on fire.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum


class PsychologicalDemon(Enum):
    """The 10 demons that kill vol traders"""
    GAMMA_PANIC = "gamma_panic"
    FOMO_LONG_VOL = "fomo_long_vol_bottom"
    REVENGE_SCALING = "revenge_scaling"
    HOPE_CREEP = "hope_creep"
    EUPHORIA_OVERSIZE = "euphoria_after_winstreak"
    BARGAIN_HUNT_SPIKE = "bargain_hunt_vol_spike"
    REGIME_PARALYSIS = "regime_change_paralysis"
    CHEAP_IV_OVERCONFIDENCE = "cheap_iv_overconfidence"
    JOURNAL_THEATER = "journal_theater"
    LIFESTYLE_CREEP = "lifestyle_creep_from_theta"


@dataclass
class DemonProfile:
    """Profile of a psychological demon"""
    name: str
    what_it_feels_like: str
    why_it_kills: str
    permanent_fix: str
    implementation_steps: List[str]
    severity: int  # 1-10


# ========================================
# DEMON PROFILES
# ========================================

DEMON_PROFILES = {
    PsychologicalDemon.GAMMA_PANIC: DemonProfile(
        name="Gamma Panic",
        what_it_feels_like="You're short gamma, market rips 3% in 20 minutes, P&L is down 8-15% "
                           "in an hour, heart rate 150 bpm",
        why_it_kills="Override all stops, add to losers, revenge trade, blow up account",
        permanent_fix="Pre-commit to the 21-DTE rule and 2× credit stop. Write the exact dollar "
                     "stop on a physical card taped to your monitor. When it hits, you close—no debate.",
        implementation_steps=[
            "Create physical stop-loss card with exact dollar amounts",
            "Tape to monitor before opening any position",
            "Set automatic alerts at stop levels",
            "Close screen and walk away for 30 min after a stop",
            "Never override a mechanical stop",
        ],
        severity=10,
    ),

    PsychologicalDemon.FOMO_LONG_VOL: DemonProfile(
        name="FOMO Long Vol at Bottom",
        what_it_feels_like="VIX at 12, everything calm, you 'know' a crash is coming so you buy "
                           "cheap OTM VIX calls that bleed for 9 months",
        why_it_kills="Continuous theta bleed, hope turns to despair, miss the actual spike",
        permanent_fix="Force yourself to buy long vol only when IV Rank <25% AND you have a defined "
                     "catalyst. Otherwise you are allowed exactly one tiny VIX butterfly per quarter "
                     "as emotional masturbation—nothing more.",
        implementation_steps=[
            "Only buy long vol when IV Rank <25%",
            "Must identify specific catalyst within 30 days",
            "Quarterly VIX butterfly allowance: 0.25% of account max",
            "Track VIX butterfly entries in separate log",
            "No other long vol trades without catalyst",
        ],
        severity=8,
    ),

    PsychologicalDemon.REVENGE_SCALING: DemonProfile(
        name="Revenge Scaling After a Loser",
        what_it_feels_like="Just got stopped on a short strangle, immediately double size on the "
                           "next one 'to get it back'",
        why_it_kills="Compound losses, emotional trading, violate risk management, blow up",
        permanent_fix="One trade per day max after a mechanical stop-loss exit. Mandate a 24-hour "
                     "cooling-off period. Pros literally lock their platform with a time-lock safe "
                     "or use API rate limiting.",
        implementation_steps=[
            "After stop loss: 24-hour mandatory break",
            "Set calendar reminder before next trade",
            "Use broker API to limit trades per day",
            "Physical lock box for hardware keys (extreme)",
            "Write 'WHY' you want to enter next trade",
        ],
        severity=9,
    ),

    PsychologicalDemon.HOPE_CREEP: DemonProfile(
        name="Hope Creep on Tested Short Premium",
        what_it_feels_like="Position down 150% of credit, 'it always comes back,' you hold into "
                           "expiration week",
        why_it_kills="Turn small loss into account-destroying loss, gamma explosion, assignment disaster",
        permanent_fix="Hard rule: if buy-to-close ever costs >2.1× credit received → close immediately. "
                     "No exceptions, no 'but this time NVDA earnings will save me.'",
        implementation_steps=[
            "Set alert at 2.0× credit received",
            "Emergency exit at 2.1× (no discretion)",
            "Review every hope trade in journal",
            "Calculate cost of hope (actual vs stop loss)",
            "Never hold tested position past 14 DTE",
        ],
        severity=10,
    ),

    PsychologicalDemon.EUPHORIA_OVERSIZE: DemonProfile(
        name="Euphoria After Win Streak",
        what_it_feels_like="12 winning short straddles in a row, you start selling 0DTE naked "
                           "strangles with 5% of account",
        why_it_kills="One VIX spike wipes out 10 wins, overconfidence right before disaster",
        permanent_fix="Mandatory 50% profit withdrawal or forced two-week vacation after every 10 "
                     "consecutive winners. Your brain on a win streak is chemically identical to cocaine. "
                     "Treat it as such.",
        implementation_steps=[
            "Track win streaks in trading journal",
            "After 10 wins: withdraw 50% of profits",
            "OR take 2-week forced vacation",
            "Reduce position size by 50% during streak",
            "No new strategy types during win streaks",
        ],
        severity=9,
    ),

    PsychologicalDemon.BARGAIN_HUNT_SPIKE: DemonProfile(
        name="Bargain Hunting Long Vol After Spike",
        what_it_feels_like="VIX spikes to 45, you buy ATM straddles 'because it's going higher' → "
                           "instant vol crush",
        why_it_kills="Buy high, vol crush, theta decay, loss on 'obvious' trade",
        permanent_fix="Long vol only on the way up, never chase the spike. The only permissible "
                     "post-spike long-vol trade is a tiny ratio backspread entered the day VIX "
                     "closes red after a 40+ day.",
        implementation_steps=[
            "Never buy long vol on spike days",
            "Wait for VIX red day after >40 spike",
            "Only ratio backspreads post-spike",
            "Maximum 1% account risk on post-spike trades",
            "Set 2-day timer before any long vol entry",
        ],
        severity=7,
    ),

    PsychologicalDemon.REGIME_PARALYSIS: DemonProfile(
        name="Paralysis on Regime Change",
        what_it_feels_like="VIX jumps from 18 → 32 overnight, you freeze and watch short vega bleed",
        why_it_kills="Know what to do but can't act, watch disaster unfold, paralysis = loss",
        permanent_fix="Pre-written Regime Kill Switch: if VIX closes >30 → everything short vol is "
                     "closed by next open, no questions. Script it or give the password to a "
                     "spouse/friend.",
        implementation_steps=[
            "Pre-write regime exit orders",
            "Set VIX alerts at 25, 30, 35, 40",
            "Automatic email/SMS at VIX >30",
            "Give spouse/friend 'kill switch' password",
            "Practice regime transition scenarios monthly",
        ],
        severity=10,
    ),

    PsychologicalDemon.CHEAP_IV_OVERCONFIDENCE: DemonProfile(
        name="Over-Confidence in 'Cheap' IV",
        what_it_feels_like="IV Rank 18%, feels safe to sell iron condors for $1.20 credit → "
                           "October 1987, Aug 2011, March 2020, etc.",
        why_it_kills="Low IV = low premium = low margin for error, blow up when impossible happens",
        permanent_fix="Never forget: low IV Rank = low premium = low margin for error. Cap short-vol "
                     "risk at 1% of account when IV Rank <40% instead of the usual 3-4%.",
        implementation_steps=[
            "Max 1% risk when IV Rank <40%",
            "Require 2× normal edge for low IV trades",
            "Reduce position size by 50% in R1",
            "No undefined risk when IV Rank <30%",
            "Track low-IV performance separately",
        ],
        severity=8,
    ),

    PsychologicalDemon.JOURNAL_THEATER: DemonProfile(
        name="Journal Theatre",
        what_it_feels_like="You keep a beautiful journal but never read the old entries",
        why_it_kills="Repeat same mistakes, no actual learning, false sense of discipline",
        permanent_fix="Quarterly mandatory 'Losers Anonymous' review: read every single losing trade "
                     "from the past 12 months out loud (or to an accountability partner). Forces "
                     "pattern recognition.",
        implementation_steps=[
            "Set quarterly review date (non-negotiable)",
            "Read ALL losing trades aloud",
            "Identify top 3 recurring mistakes",
            "Update trading rules based on patterns",
            "Find accountability partner for review",
        ],
        severity=6,
    ),

    PsychologicalDemon.LIFESTYLE_CREEP: DemonProfile(
        name="Lifestyle Creep from Theta",
        what_it_feels_like="Making $4k/week collecting premium → buying cars, bigger house, raising "
                           "monthly burn",
        why_it_kills="Need theta to live, can't stop trading, forced to trade in bad conditions, blow up",
        permanent_fix="Separate 'theta income' account. Only living expenses come from a salary job "
                     "or fixed withdrawal schedule. Everything else stays in the war chest. Most "
                     "blown-up vol traders died from lifestyle, not delta.",
        implementation_steps=[
            "Open separate income account",
            "Fixed monthly withdrawal only",
            "Theta profits stay in trading account",
            "Build 12-month expense reserve",
            "Never increase lifestyle from trading gains",
        ],
        severity=9,
    ),
}


class PsychologicalGuardrails:
    """
    System to track and prevent psychological trading errors
    """

    def __init__(self):
        self.stop_loss_hits: List[datetime] = []
        self.consecutive_wins: int = 0
        self.consecutive_losses: int = 0
        self.last_regime_check: Optional[datetime] = None
        self.hope_creep_warnings: int = 0
        self.euphoria_alerts: int = 0

    def check_revenge_trading_cooldown(self) -> Tuple[bool, str]:
        """
        Check if in mandatory 24-hour cooldown after stop loss

        Returns:
            (can_trade, message)
        """
        if not self.stop_loss_hits:
            return True, "No recent stop losses"

        last_stop = self.stop_loss_hits[-1]
        hours_since_stop = (datetime.now() - last_stop).total_seconds() / 3600

        if hours_since_stop < 24:
            return False, f"⚠️ REVENGE TRADING COOLDOWN: {24 - hours_since_stop:.1f} hours remaining"

        return True, "Cooldown period complete"

    def record_stop_loss(self) -> None:
        """Record a stop loss hit and trigger cooldown"""
        self.stop_loss_hits.append(datetime.now())
        self.consecutive_wins = 0

    def record_win(self) -> Optional[str]:
        """
        Record a winning trade and check for euphoria warning

        Returns:
            Warning message if euphoria detected
        """
        self.consecutive_wins += 1
        self.consecutive_losses = 0

        # Euphoria warning after 10 consecutive wins
        if self.consecutive_wins >= 10:
            self.euphoria_alerts += 1
            return (
                f"🚨 EUPHORIA ALERT: {self.consecutive_wins} consecutive wins!\n"
                f"MANDATORY ACTION: Withdraw 50% of profits OR take 2-week vacation.\n"
                f"Your brain is on cocaine right now. Treat it as such."
            )

        # Mild warning at 7 wins
        if self.consecutive_wins >= 7:
            return (
                f"⚠️ Win Streak Alert: {self.consecutive_wins} consecutive wins.\n"
                f"Consider reducing position size by 25-50%."
            )

        return None

    def record_loss(self) -> None:
        """Record a losing trade"""
        self.consecutive_losses += 1
        self.consecutive_wins = 0

    def check_hope_creep(
        self,
        entry_credit: float,
        current_buyback_cost: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if position is in hope creep territory

        Args:
            entry_credit: Original credit received
            current_buyback_cost: Current cost to buy back

        Returns:
            (is_hope_creep, warning_message)
        """
        if entry_credit == 0:
            return False, None

        loss_multiple = current_buyback_cost / entry_credit

        # Warning at 1.5×
        if loss_multiple >= 1.5:
            warning = (
                f"⚠️ Hope Creep Warning: Position at {loss_multiple:.2f}× credit received\n"
                f"Current loss: ${(current_buyback_cost - entry_credit) * 100:,.2f}\n"
                f"Approaching 2.1× emergency exit threshold"
            )
            return False, warning

        # Emergency at 2.1×
        if loss_multiple >= 2.1:
            self.hope_creep_warnings += 1
            emergency = (
                f"🚨 HOPE CREEP EMERGENCY: Position at {loss_multiple:.2f}× credit received!\n"
                f"CLOSE IMMEDIATELY - NO EXCEPTIONS\n"
                f"Current loss: ${(current_buyback_cost - entry_credit) * 100:,.2f}"
            )
            return True, emergency

        return False, None

    def check_regime_paralysis(
        self,
        current_vix: float,
        has_short_vol_positions: bool,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check for regime change paralysis

        Returns:
            (should_act, message)
        """
        # Kill switch at VIX >30
        if current_vix > 30 and has_short_vol_positions:
            return True, (
                f"🚨 REGIME KILL SWITCH ACTIVATED: VIX = {current_vix:.2f}\n"
                f"CLOSE ALL SHORT VOL POSITIONS BY NEXT OPEN\n"
                f"No discretion. No questions. This is pre-programmed."
            )

        # Warning at VIX >25
        if current_vix > 25 and has_short_vol_positions:
            return False, (
                f"⚠️ Regime Warning: VIX = {current_vix:.2f}\n"
                f"Review short vol positions. Prepare for regime transition."
            )

        return False, None

    def check_cheap_iv_overconfidence(
        self,
        iv_rank: float,
        position_risk_pct: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if oversizing in low IV environment

        Returns:
            (is_violation, warning)
        """
        # Low IV requires lower risk
        if iv_rank < 40:
            max_risk = 0.01  # 1% max in low IV
            if position_risk_pct > max_risk:
                return True, (
                    f"🚨 CHEAP IV OVERCONFIDENCE: IV Rank {iv_rank:.1f}% but risking "
                    f"{position_risk_pct*100:.1f}%\n"
                    f"Maximum allowed: {max_risk*100}% when IV Rank <40%\n"
                    f"Reduce position size by {((position_risk_pct - max_risk) / position_risk_pct) * 100:.0f}%"
                )

        return False, None

    def get_psychological_status(self) -> Dict[str, any]:
        """Get current psychological status summary"""
        can_trade, cooldown_msg = self.check_revenge_trading_cooldown()

        return {
            'can_trade': can_trade,
            'cooldown_message': cooldown_msg if not can_trade else None,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'total_stop_losses': len(self.stop_loss_hits),
            'hope_creep_warnings': self.hope_creep_warnings,
            'euphoria_alerts': self.euphoria_alerts,
            'warnings': self._get_active_warnings(),
        }

    def _get_active_warnings(self) -> List[str]:
        """Get list of active psychological warnings"""
        warnings = []

        if self.consecutive_wins >= 7:
            warnings.append(f"Win streak: {self.consecutive_wins} trades")

        if self.consecutive_losses >= 3:
            warnings.append(f"Loss streak: {self.consecutive_losses} trades")

        recent_stops = [s for s in self.stop_loss_hits if (datetime.now() - s).days <= 7]
        if len(recent_stops) >= 3:
            warnings.append(f"{len(recent_stops)} stop losses in past 7 days")

        return warnings


# The Only Three Psychological Rules That Matter
GOLDEN_RULES = """
THE ONLY THREE PSYCHOLOGICAL RULES THAT MATTER
==============================================

1. Never risk money you have already spent in your head.
   If you're mentally spending next month's theta before it's collected,
   you will override every stop.

2. Your first loss is your best loss—always.
   A $2,800 stop today prevents a $48,000 expiration-week disaster tomorrow.

3. Trade like the market is trying to bankrupt you personally.
   Because on some days, it really is.

Master the psychology and the math takes care of itself.
Fail the psychology and even perfect edges turn into donation accounts.
"""


def get_demon_fix(demon: PsychologicalDemon) -> DemonProfile:
    """Get the fix for a specific psychological demon"""
    return DEMON_PROFILES[demon]


def get_all_demons() -> Dict[PsychologicalDemon, DemonProfile]:
    """Get all demon profiles"""
    return DEMON_PROFILES
