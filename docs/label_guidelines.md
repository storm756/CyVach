# CyVach label guidelines (v1, 2026-10-17; taxonomy = CLAUDE.md §5 default, accepted at Gate 1)

Four labels. The names are fixed for app compatibility; in the paper, `spam` means **scam/fraud**.
All examples below are **invented for illustration**. None is a real message.

## Decision procedure (apply in order)

1. Does the message try to **defraud, steal credentials or money, or get malware installed**, or ask you to **share or forward an OTP/PIN**? → **spam**
2. Otherwise, is it a **legitimate security-sensitive** message (OTP or verification code, login/new-device alert, password/PIN change, unusual-activity alert)? → **caution**
3. Otherwise, is it **commercial marketing** (offers, discounts, sales, brand announcements, plan upsells), solicited or not? → **promo**
4. Otherwise (person-to-person chat, non-sensitive notifications: debit/credit/balance, delivery status, bill paid, tickets, reminders) → **ham**

**Tie-break:** the highest severity wins, `spam > caution > promo > ham`. Set **borderline = 1** if you hesitated between two labels after applying the rules.
Judge the **text** (and the sender, if shown). Do not guess at facts you cannot see. A link alone does not make a message spam; a link plus pressure, an impersonated brand, or a request for credentials or payment does.

## ham
- "Reached home, will call you after dinner."
- "Rs 1,250.00 debited from A/c XX4821 on 12-Oct to SWIGGY. Avl bal Rs 8,402.17."
- "Your order #40213 has been delivered. Thank you for shopping with us."
- "PNR 4521873: Train 12951 Coach B3 Berth 41 confirmed."
- "Reminder: your electricity bill of Rs 860 is due on 25-Oct." (a plain reminder with no threat and no link to pay elsewhere)
- "Payment of Rs 499 received for your broadband account. Thank you."

## caution
- "482913 is your OTP for login to XYZ Bank NetBanking. Do not share it with anyone."
- "New login to your account from Chrome on Windows, Pune. Not you? Contact support."
- "WhatsApp Web is now active on a new device."
- "Your UPI PIN was changed successfully. If not done by you, call your bank."
- "Unusual activity detected on your credit card ending 1122. Call the number on the back of your card."
- "Your Aadhaar-linked mobile verification code is 739204. Valid for 10 minutes."

## spam (scam/fraud)
- "Dear customer, your account KYC has expired and will be blocked today. Update now: http://kyc-update-bank.xyz"
- "Congratulations! You won Rs 25 lakh in the KBC lucky draw. Pay Rs 4,999 processing fee to claim."
- "Part-time job: earn Rs 5,000/day liking YouTube videos. WhatsApp now."
- "Your parcel is held at customs. Pay Rs 150 to release: bit.ly/xxxx"
- "Electricity will be disconnected tonight at 9:30 pm. Call officer 98xxxxxxxx immediately."
- "I sent an OTP to your number by mistake, please forward it to me." (OTP-sharing request)
- An **OTP-style text from a personal mobile number** → spam.

## promo
- "Big Billion Days start tomorrow! Up to 80% off on mobiles."
- "Recharge with Rs 299 and get 2GB/day + Disney+ for 3 months."
- "Flat 50% off on your next order with code TASTY50."
- "Tiffin service available: home-made food, monthly plans. Call to subscribe." (unsolicited classified ad, not deceptive)
- "Get your free credit report today. Download our app."

## Edge cases and rules

| Situation | Label | Rule |
|---|---|---|
| Promo with an unbelievable prize or advance fee ("free iPhone, pay shipping") | spam | Deception beats marketing |
| Real bank OTP that *warns* "do not share" | caution | The warning text does not make it spam |
| Scam that *adds* "do not share your OTP" as camouflage | spam | Judge the intent |
| Missed-call / "call back" alert from the operator | ham | Non-sensitive notification |
| Loan offer from a registered lender with no deception | promo | Not caution |
| Instant-loan, "pre-approved" offer from a personal number, link to an APK | spam | Malware or fraud signals |
| Bank fraud-awareness advisory ("Beware of KYC frauds") | ham | Informational, not security-sensitive about *your* account |
| Transaction alert containing "not you? call …" | caution if it is a security alert; ham if it is a plain debit alert with a standard footer | Is the primary purpose security? |
| Charity or blood-donation appeal from a person | ham | Not commercial, not fraud |
| Survey or feedback request from an operator | promo | Brand notice |
| Message in Hindi/Marathi/Gujarati or Hinglish | same rules | Label the meaning |
| Scrubbed placeholders `<NAME>`, `<PHONE>`, `<URL>` | ignore them | They are artifacts of the scrubber |
| Cannot tell legitimate from fake (e.g. a bare "Update your app: <URL>") | best guess + borderline = 1 | |
