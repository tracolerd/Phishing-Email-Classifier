"""
data/generate_mock_data.py

Programmatically generates a synthetic phishing email dataset.
Intentional class overlap and per-sample noise are included to prevent overfitting.

Output:  data/mock_emails.csv
Columns: subject | body | has_attachments | url_count | spf_fail | label
         label → 1 = Phishing, 0 = Legitimate
"""

import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
N_SAMPLES: int = 1200
PHISHING_RATIO: float = 0.50            # balanced dataset
OUTPUT_PATH: Path = Path(__file__).resolve().parent / "mock_emails.csv"

# ── Phishing Corpus ───────────────────────────────────────────────────────────
PHISHING_SUBJECTS = [
    "URGENT: Your account has been suspended",
    "Action Required: Verify your identity immediately",
    "Security Alert - Unusual login detected",
    "Your PayPal account is permanently limited",
    "Final Notice: Confirm your details or lose access",
    "Congratulations! You are our selected winner",
    "Important: Update your billing information now",
    "Your account will be closed in 24 hours",
    "Suspicious activity detected on your account",
    "Verify your email address to restore access",
    "Your package could not be delivered - Act now",
    "IRS Tax Refund: Claim your $2,340 refund today",
    "Bank Security Alert: Unauthorized transaction found",
    "Netflix: Your payment has failed",
    "Apple ID suspended - Immediate verification needed",
    "Microsoft: Your Outlook access has been blocked",
    "DHL: Failed delivery - Reschedule now",
    "HSBC: Your card has been temporarily frozen",
    "LinkedIn: Someone tried to access your account",
    "Amazon: Your account has been locked for review",
]

PHISHING_BODY_TEMPLATES = [
    "<html><body><p>Dear Customer,</p><p>We detected suspicious activity on your account. "
    "Click <a href='http://secure-verify-now.com/login'>here</a> to verify your identity. "
    "Failure to act within 24 hours will result in permanent suspension. "
    "Also visit <a href='http://account-restore.net'>account-restore.net</a>.</p></body></html>",

    "<html><body>Your account has been compromised. Restore access by providing your "
    "credentials at <a href='http://phish-domain.net/login'>this secure link</a>. "
    "Our security team is standing by. Do not delay. "
    "<a href='http://helpdesk-verify.com'>Contact support here</a>.</body></html>",

    "Congratulations! You have been selected as our lucky winner of $1,000. "
    "Claim your prize at http://fake-prize.com and enter your bank details. "
    "Offer expires in 2 hours! Visit http://prize-winner-claim.net for details.",

    "<p>URGENT: Your card was charged $499.99 for an unrecognized purchase. "
    "If not you, <a href='http://refund-claim.biz'>click here to cancel</a>. "
    "For additional help: http://support-refund.co/cancel</p>",

    "Your mailbox is 99% full. Upgrade for free at http://mail-storage-upgrade.net "
    "before emails stop delivering. Act now: http://mail-upgrade-free.com/claim. "
    "Limited time offer expires soon.",

    "<html><body>Your tax refund of $3,200 is pending approval. "
    "Submit your SSN and banking details at <a href='http://irs-refund-portal.com'>this portal</a>. "
    "Complete within 48 hours. <a href='http://irs-secure-claim.net'>Claim refund</a>"
    "</body></html>",

    "We could not deliver your Amazon package. Update your address and pay a $1.99 fee at "
    "http://delivery-reschedule.co. Reference: PKG-88234. "
    "Track at http://tracking-update.biz/88234",

    "<html><body>Your Netflix membership has been suspended due to a billing issue. "
    "<a href='http://netflix-billing-update.com'>Update payment info</a> to continue. "
    "Or visit http://netfllx-secure.com/account to restore access.</body></html>",

    "Security breach detected on your Google account. Change your password immediately "
    "at http://google-security-alert.net. Two-factor reset: http://goo-gle-verify.com/2fa. "
    "Do not ignore this warning.",

    "<p>Your Apple ID was used from an unknown device in Russia. "
    "Verify at <a href='http://apple-id-verify.org'>this link</a> or your ID will be deleted. "
    "Help: <a href='http://apple-support-help.biz'>Contact Apple Support</a></p>",

    "HR Notice: Your salary structure has been updated this month. "
    "Review and sign the contract at http://hr-portal-staff.com/sign by end of day. "
    "Login: http://staff-hr-secure.net using your company credentials.",

    "<html><body>Dear User, We are migrating our banking systems. Please re-enter your "
    "credentials at <a href='http://bank-migration-secure.com'>secure.bank-migrate.com</a>. "
    "Without verification by tomorrow, transactions will be blocked. "
    "<a href='http://online-banking-reauth.net'>Verify now</a></body></html>",

    "Your Dropbox storage is over the limit. Files will be deleted in 24 hours unless you "
    "upgrade for free at http://dropbox-free-upgrade.com. "
    "Login: http://dropboxs-secure.net/upgrade",

    "Payroll encountered an error with your direct deposit. Submit updated banking info at "
    "http://payroll-update-portal.org immediately to avoid a delayed paycheck. Urgent.",

    "<html><body>You have a pending wire transfer of $5,000 waiting for your approval. "
    "Confirm via <a href='http://wire-transfer-secure.net'>secure wire portal</a>. "
    "This expires in 3 hours: <a href='http://wires-bank-confirm.com'>Confirm now</a>"
    "</body></html>",

    "Your LinkedIn account will be restricted due to suspicious sign-in. Verify now at "
    "http://linkedin-verify-account.net or contact us at http://linke-din-support.com/help",

    "<html><body>FINAL WARNING: Your Microsoft 365 subscription has expired. "
    "Renew at <a href='http://ms365-renew-now.com'>this link</a> or lose access to all files. "
    "<a href='http://microsoft-billing-secure.net'>Renew subscription</a></body></html>",

    "Your DocuSign document is ready to sign. Access it at http://docu-sign-secure.org/doc "
    "before it expires in 24 hours. Document ID: DS-44821. Sign: http://sign-docu-now.com",

    "<p>Dear Account Holder, Your internet banking access has been revoked. "
    "Reinstate access at <a href='http://online-bank-reinstate.com'>this secure page</a>. "
    "Call our hotline: <a href='http://bank-hotline-secure.net'>1-800-BANK-NOW</a></p>",

    "ALERT: Your antivirus has expired. Your PC is at risk. Renew now at "
    "http://antivirus-renew-now.com for only $1.99. "
    "Download protection: http://av-protect-now.net/install",
]

# ── Legitimate Corpus ─────────────────────────────────────────────────────────
LEGITIMATE_SUBJECTS = [
    "Team sync tomorrow at 10 AM - agenda inside",
    "Q3 project status update for your review",
    "Your invoice #4521 is now available",
    "Welcome to our monthly newsletter",
    "Meeting notes from today's product call",
    "Reminder: Submit your timesheet by Friday",
    "Your order has shipped - tracking info enclosed",
    "New comment added on your shared document",
    "Weekly digest: Top industry stories",
    "Re: Follow-up from our last discussion",
    "Your appointment is confirmed",
    "Project deadline updated - please read",
    "Introducing our newest team member",
    "Annual subscription renewal reminder",
    "We received your feedback - thank you!",
    "Action items from yesterday's standup",
    "Design review scheduled for Thursday at 2 PM",
    "Q4 planning kickoff - save the date",
    "Your certificate of completion is ready",
    "Server maintenance scheduled for Sunday 2 AM",
]

LEGITIMATE_BODY_TEMPLATES = [
    "Hi team, just a reminder that our weekly sync is tomorrow at 10 AM. "
    "Please review the agenda shared in the project folder and come prepared with updates. "
    "The Zoom link is in the calendar invite.",

    "Dear valued customer, thank you for your purchase. Your order #78432 has been shipped "
    "and is expected to arrive within 3-5 business days. "
    "Track your shipment from your account dashboard.",

    "Hi everyone, please find the meeting notes from today's all-hands meeting attached. "
    "Review the action items assigned to you and update the tracker accordingly. "
    "Reach out to your team lead with any questions.",

    "Hello, your invoice for services rendered in October is now available in the billing portal. "
    "Total amount due: $250.00, payment due by the 15th. "
    "Contact billing@company.com with any questions.",

    "Good morning! I wanted to follow up on our Q4 roadmap discussion. "
    "Can we schedule a 30-minute call this week to align on priorities? "
    "I am free Tuesday or Wednesday afternoon.",

    "Your Premium Plan will auto-renew on December 1st. No action required. "
    "To cancel or make changes, visit your account settings page. "
    "Thank you for being a loyal customer.",

    "This is a friendly reminder to submit your timesheet for the current pay period "
    "by end of business Friday to ensure timely payroll processing. "
    "Contact HR if you experience issues with the portal.",

    "We are pleased to introduce Sarah Johnson, who joins the engineering team "
    "as Senior Developer starting Monday. Give her a warm welcome! "
    "Her email is sarah.j@company.com.",

    "Here is your weekly roundup of top stories in tech, product updates, "
    "and community highlights from the past seven days. "
    "Reply to this email to unsubscribe from future editions.",

    "Thank you for submitting feedback on the beta release. Your suggestions have been logged "
    "and will be reviewed during the next sprint planning session. "
    "We truly appreciate your time and input.",

    "The project deadline has been moved from November 10 to November 17 to accommodate "
    "the additional client review cycle requested last week. "
    "Please update your schedules and inform external stakeholders accordingly.",

    "Hey, just checking in on the status of the homepage redesign mockups. "
    "Let us know if you need additional assets or if there are any blockers. "
    "We need them finalized before Thursday's review session.",

    "Your monthly account statement for October is now available. "
    "Log in to view transactions and download a PDF copy for your records. "
    "Call customer service at 1-800-555-0100 for any questions.",

    "The Q3 business review slide deck has been updated with the latest metrics. "
    "Please review it before Thursday's leadership meeting. "
    "Comments are enabled on the shared document.",

    "Your certificate of completion for the Python Advanced course is ready for download. "
    "Access it from your learning dashboard. "
    "Congratulations on completing the program!",

    "Planned maintenance on production servers is scheduled this Sunday from 2 AM to 4 AM UTC. "
    "Expect brief service interruptions during that window. "
    "Contact devops@company.com if you have concerns.",

    "Here is the agenda for Thursday's design review at 2 PM. "
    "We will cover the new onboarding flow and dashboard redesign. "
    "Please come prepared with any open feedback or questions.",

    "As we close out Q3, I want to thank everyone for their hard work and dedication. "
    "The results speak for themselves - 15% above our revenue target. "
    "See you all at the team lunch on Friday!",

    "Your request to reset your company password has been received. "
    "IT will process this within one business day. "
    "If you did not submit this request, please contact the helpdesk immediately.",

    "The annual performance review cycle begins next Monday. "
    "You will receive a self-assessment form via HR portal by end of this week. "
    "Please complete it within 10 business days.",
]

# ── Subject variation prefixes for realism ────────────────────────────────────
PHISHING_PREFIXES = ["RE: ", "FWD: ", "[Action Required] ", "⚠ ", "[EXTERNAL] "]
LEGIT_PREFIXES = ["RE: ", "FWD: ", "[Team] ", "Update: "]


def _generate_phishing_sample() -> Dict:
    """
    Generate a single synthetic phishing email record.

    Phishing characteristics:
    - Higher url_count (3–10, skewed toward middle range)
    - has_attachments biased toward 1
    - spf_fail biased toward 1 (malicious senders rarely pass SPF)
    """
    subject = random.choice(PHISHING_SUBJECTS)
    body = random.choice(PHISHING_BODY_TEMPLATES)

    # Vary subject prefix with 25% probability to increase diversity
    if random.random() < 0.25:
        subject = random.choice(PHISHING_PREFIXES) + subject

    url_count = int(np.random.choice(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        p=[0.05, 0.10, 0.15, 0.20, 0.15, 0.12, 0.10, 0.07, 0.04, 0.02],
    ))
    has_attachments = int(np.random.choice([0, 1], p=[0.30, 0.70]))
    spf_fail = int(np.random.choice([0, 1], p=[0.20, 0.80]))

    return {
        "subject": subject,
        "body": body,
        "has_attachments": has_attachments,
        "url_count": url_count,
        "spf_fail": spf_fail,
        "label": 1,
    }


def _generate_legitimate_sample() -> Dict:
    """
    Generate a single synthetic legitimate email record.

    Legitimate characteristics:
    - Lower url_count (0–3 typically)
    - Exception: newsletters may have 3–7 links (intentional overlap with phishing)
    - spf_fail almost always 0
    """
    subject = random.choice(LEGITIMATE_SUBJECTS)
    body = random.choice(LEGITIMATE_BODY_TEMPLATES)

    if random.random() < 0.15:
        subject = random.choice(LEGIT_PREFIXES) + subject

    # 15% chance of high-link newsletters for realistic class overlap
    is_newsletter = random.random() < 0.15
    if is_newsletter:
        url_count = int(np.random.randint(3, 8))
    else:
        url_count = int(np.random.choice(
            [0, 1, 2, 3, 4, 5],
            p=[0.30, 0.35, 0.20, 0.08, 0.04, 0.03],
        ))

    has_attachments = int(np.random.choice([0, 1], p=[0.60, 0.40]))
    spf_fail = int(np.random.choice([0, 1], p=[0.93, 0.07]))

    return {
        "subject": subject,
        "body": body,
        "has_attachments": has_attachments,
        "url_count": url_count,
        "spf_fail": spf_fail,
        "label": 0,
    }


def generate_dataset(n_samples: int, phishing_ratio: float) -> pd.DataFrame:
    """
    Build a balanced synthetic email dataset.

    Args:
        n_samples:       Total number of records to generate.
        phishing_ratio:  Fraction of records that are phishing (0.0 – 1.0).

    Returns:
        Shuffled DataFrame with columns:
        ['subject', 'body', 'has_attachments', 'url_count', 'spf_fail', 'label']
    """
    n_phishing = int(n_samples * phishing_ratio)
    n_legitimate = n_samples - n_phishing

    records = (
        [_generate_phishing_sample()  for _ in range(n_phishing)]
        + [_generate_legitimate_sample() for _ in range(n_legitimate)]
    )

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Enforce correct dtypes
    df["has_attachments"] = df["has_attachments"].astype(int)
    df["url_count"]       = df["url_count"].astype(int)
    df["spf_fail"]        = df["spf_fail"].astype(int)
    df["label"]           = df["label"].astype(int)

    return df


def main() -> None:
    """Entry point: generate dataset and persist to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Generating {N_SAMPLES} synthetic email records ...")
    df = generate_dataset(N_SAMPLES, PHISHING_RATIO)

    df.to_csv(OUTPUT_PATH, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    phishing_count   = int(df["label"].sum())
    legitimate_count = int((df["label"] == 0).sum())

    print(f"[+] Dataset saved  → {OUTPUT_PATH}")
    print(f"\n{'='*54}")
    print(f"  Total records   : {len(df)}")
    print(f"  Phishing  (1)   : {phishing_count}  ({phishing_count / len(df) * 100:.1f}%)")
    print(f"  Legitimate (0)  : {legitimate_count} ({legitimate_count / len(df) * 100:.1f}%)")
    print(f"{'='*54}")
    print(f"\n  Mean Feature Values by Class:")
    summary = df.groupby("label")[["url_count", "has_attachments", "spf_fail"]].mean()
    summary.index = ["Legitimate (0)", "Phishing (1)"]
    print(summary.round(3).to_string())
    print(f"\n{'='*54}\n")


if __name__ == "__main__":
    main()
