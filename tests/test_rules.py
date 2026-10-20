"""Hand-checked tests for src/rules.py against the shipped TypeScript (SMSReader.tsx, scamDetector.ts).

Each expected value was derived by reading the TS source line by line (see docs/rule_layer_as_shipped.md).
Run:  .venv/Scripts/python.exe -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rules import ALL_LAYERS, final_classification as fc, is_mobile_number, pipeline, pre_model_override  # noqa: E402


class PreModel(unittest.TestCase):
    def test_h1_missed_call_jio(self):
        r = pre_model_override("You have a missed call from 98xxxxxxxx. Team Jio")
        self.assertEqual((r.label, r.confidence), ("ham", 1.0))

    def test_h1_vi_substring_matches_common_words(self):
        # 'vi' is a substring of 'service' -> operator check passes (as in the TS code)
        self.assertEqual(pre_model_override("Last missed call at 10:00. Call service centre").label, "ham")

    def test_h1_needs_operator(self):
        self.assertIsNone(pre_model_override("missed call from mom"))

    def test_h2_utility(self):
        r = pre_model_override("Payment received for your Airtel bill. Thank you")
        self.assertEqual((r.label, r.confidence), ("ham", 0.98))

    def test_h2_needs_provider(self):
        self.assertIsNone(pre_model_override("Your payment received successfully"))

    def test_pre_layer_off(self):
        self.assertIsNone(pre_model_override("missed call from x jio", layers=frozenset()))


class Trusted(unittest.TestCase):
    def test_trusted_vetoes_spam(self):
        self.assertEqual(fc("VM-HDFCBK-S", "KYC expired click link", "spam", 0.9999).label, "ham")

    def test_trusted_vetoes_caution_otp(self):
        r = fc("AD-HDFCBK-S", "123456 is your OTP", "caution", 0.99)
        self.assertEqual((r.label, r.confidence), ("ham", 0.99))

    def test_trusted_keeps_promo(self):
        r = fc("JM-AIRTEL-P", "recharge offer", "promo", 0.8)
        self.assertEqual((r.label, r.confidence), ("promo", 0.8))

    def test_trusted_substring_vi(self):
        # 'VI' inside 'VM-VISA-S' -> treated as trusted (substring match, shipped behaviour)
        self.assertEqual(fc("VM-VISA-S", "Canada visa guaranteed", "spam", 0.99).rule, "R0_trusted_veto")

    def test_trusted_case_insensitive_sender(self):
        self.assertEqual(fc("jm-swiggy-s", "your order", "caution", 0.6).label, "ham")


class Safety(unittest.TestCase):
    def test_safety_vetoes_spam(self):
        self.assertEqual(fc("AB-RANDOM", "Do not share your OTP. Verify KYC at link", "spam", 0.99).label, "ham")

    def test_safety_ignores_non_spam(self):
        self.assertEqual(fc("AB-RANDOM", "Security alert: new login", "caution", 0.9).label, "caution")


class Mobile(unittest.TestCase):
    def test_mobile_regex_variants(self):
        for a in ("9876543210", "+919876543210", "91 98765 43210", "09876543210", "98765-43210"):
            self.assertTrue(is_mobile_number(a), a)
        for a in ("5876543210", "12345", "AD-HDFCBK", "+1 2025550101"):
            self.assertFalse(is_mobile_number(a), a)

    def test_r2_gate_spam_below_threshold(self):
        r = fc("9876543210", "hello how are you", "spam", 0.98)
        self.assertEqual((r.label, r.confidence, r.rule), ("caution", 0.98, "R2_mobile_gate"))

    def test_r2_not_when_confident(self):
        # spam conf >= 0.985 skips R2; no keyword -> falls through to model
        self.assertEqual(fc("9876543210", "hello", "spam", 0.99).label, "spam")

    def test_r3_keyword_forces_spam_even_for_ham(self):
        r = fc("9876543210", "share the otp pls", "ham", 0.99)
        self.assertEqual((r.label, r.confidence), ("spam", 1.0))

    def test_r3_substring_pin_in_shopping(self):
        # 'pin' is a substring of 'shopping' -> forced to spam (shipped behaviour)
        self.assertEqual(fc("9876543210", "going shopping later", "ham", 0.9).rule, "R3_mobile_keyword")

    def test_r2_before_r3(self):
        # spam < 0.985 from mobile with keyword: R2 fires first -> caution, not spam
        self.assertEqual(fc("9876543210", "verify your account", "spam", 0.5).label, "caution")


class Suffix(unittest.TestCase):
    def test_p_suffix_forces_promo(self):
        self.assertEqual(fc("AB-SHOPXX-P", "you won a prize", "spam", 0.9999).label, "promo")

    def test_g_suffix_forces_ham(self):
        self.assertEqual(fc("AB-NDMAEW-G", "heavy rain alert", "spam", 0.99).label, "ham")

    def test_st_downgrade_below_0999(self):
        self.assertEqual(fc("AB-SHOPXX-S", "x", "spam", 0.998).label, "ham")

    def test_st_keeps_very_confident_spam(self):
        self.assertEqual(fc("AB-SHOPXX-T", "x", "spam", 0.9995).label, "spam")

    def test_st_only_affects_spam(self):
        self.assertEqual(fc("AB-SHOPXX-S", "x", "caution", 0.5).label, "caution")

    def test_space_p_variant(self):
        self.assertEqual(fc("SHOPXX P", "x", "ham", 0.9).label, "promo")


class Pipeline(unittest.TestCase):
    def test_no_rules_returns_model(self):
        r = pipeline("AB-HDFCBK-S", "missed call from jio", lambda t: ("spam", 0.7), layers=frozenset())
        self.assertEqual((r.label, r.rule), ("spam", "model"))

    def test_pre_then_post(self):
        # H1 gives ham; mobile sender + keyword 'account'? no -> stays ham, rule reported as H1
        r = pipeline("AB-RANDOM", "You have a missed call from x. Team Jio", lambda t: ("spam", 0.9))
        self.assertEqual((r.label, r.rule), ("ham", "H1_missed_call"))

    def test_trusted_nc_variant_keeps_caution(self):
        layers = (ALL_LAYERS - {"trusted"}) | {"trusted_nc"}
        r = fc("AX-HDFCBK-S", "123456 is your OTP", "caution", 0.9, layers)
        self.assertEqual((r.label, r.rule), ("caution", "R0nc_trusted_keep_caution"))
        r = fc("AX-HDFCBK-S", "Your a/c is debited", "spam", 0.9, layers)
        self.assertEqual((r.label, r.rule), ("ham", "R0_trusted_veto"))
        r = fc("AX-HDFCBK-S", "123456 is your OTP", "caution", 0.9, ALL_LAYERS)
        self.assertEqual((r.label, r.rule), ("ham", "R0_trusted_veto"))

    def test_full_layers_constant(self):
        self.assertEqual(ALL_LAYERS, {"pre", "trusted", "safety", "mobile", "suffix"})


if __name__ == "__main__":
    unittest.main()
