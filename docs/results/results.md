# Final results (test set)

Split by website: train 27,978, validation 6,120, test 5,902.
LightGBM trees: 293. Char-CNN epochs: 24.

## Voting weight (validation F1)

| w (LightGBM) | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| F1 | 93.35 % | 93.90 % | 94.11 % | 93.37 % | 92.17 % |

Chosen: **w = 0.5**

## Test set

| Model | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| LightGBM | 89.1 % | 89.3 % | 88.9 % | 89.1 % | 10.7 % |
| Char-CNN | 91.8 % | 93.1 % | 90.4 % | 91.7 % | 6.8 % |
| Ensemble (w = 0.5) | 92.7 % | 93.9 % | 91.6 % | 92.7 % | 6.0 % |

## Risk zones on the test set (ensemble)

| Actual class | Looks safe (0-39) | Suspicious (40-69) | Phishing (70-100) |
|---|---|---|---|
| legit | 88.9 % | 8.7 % | 2.4 % |
| phishing | 5.4 % | 11.1 % | 83.5 % |

## Hand-picked tricky URLs (made up, not from the dataset)

| URL | Expected | LightGBM | CNN | Risk | Level | Note |
|---|---|---|---|---|---|---|
| `https://accounts.google.com/signin` | legit | 0.35 | 0.10 | 23 | Looks safe | real login page |
| `https://www.paypal.com/signin` | legit | 0.40 | 0.73 | 56 | Suspicious | real login page |
| `https://login.microsoftonline.com` | legit | 0.90 | 0.86 | 88 | Phishing | real login page, long domain |
| `https://github.com/login` | legit | 0.25 | 0.92 | 58 | Suspicious | real login page |
| `https://en.wikipedia.org/wiki/Phishing` | legit | 0.05 | 0.00 | 3 | Looks safe | the word phishing |
| `https://www.amazon.in/gp/your-account/order-history` | legit | 0.15 | 0.06 | 10 | Looks safe | account words, legit |
| `https://www.onlinesbi.sbi/personal/login.html` | legit | 0.88 | 0.96 | 92 | Phishing | real bank, unusual TLD |
| `http://example.com` | legit | 0.11 | 0.15 | 13 | Looks safe | plain http |
| `https://someone.github.io/portfolio` | legit | 0.16 | 0.96 | 56 | Suspicious | free hosting (known weak spot) |
| `http://paypa1-secure-login.xyz/verify` | phishing | 1.00 | 1.00 | 100 | Phishing | look-alike + cheap TLD |
| `https://secure-paypal.com.account-update.top/signin` | phishing | 1.00 | 1.00 | 100 | Phishing | brand as subdomain |
| `http://192.168.10.5/wellsfargo/login.php` | phishing | 0.99 | 1.00 | 100 | Phishing | IP address host |
| `https://rnicrosoft-office365.com/auth` | phishing | 1.00 | 0.91 | 95 | Phishing | rn -> m look-alike |
| `https://amazon-order-cancel.online/refund?id=88231` | phishing | 0.97 | 0.93 | 95 | Phishing | brand + cheap TLD |
| `https://docs-google.com.verify-user.icu/login` | phishing | 1.00 | 0.98 | 99 | Phishing | brand hidden in subdomain |
| `https://login-microsoftonline-com.web.app` | phishing | 1.00 | 1.00 | 100 | Phishing | free hosting phishing |
| `https://my-account-help.com/billing/update` | phishing | 0.87 | 0.52 | 70 | Phishing | no brand, only words |
