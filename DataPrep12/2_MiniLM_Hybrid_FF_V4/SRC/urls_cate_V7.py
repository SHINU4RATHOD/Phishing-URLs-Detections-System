from __future__ import annotations
import argparse
import csv
import logging
import concurrent.futures
import re
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import parse_qsl, unquote, urlparse

import ipaddress
import pandas as pd
try:
    import tldextract
except ImportError as exc:  # pragma: no cover - guard for missing dependency
    raise ImportError("tldextract is required. Install via 'pip install tldextract'.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "RESULTS_&_MODELS" / "1_url_cate_data10_output"


def resolve_project_path(path: Path) -> Path:
    """Resolve relative CLI paths from the project root instead of the launch cwd."""
    return path if path.is_absolute() else PROJECT_ROOT / path


# Default intelligence sets (kept outside dataclasses to avoid mutable defaults)
DEFAULT_SUSPICIOUS_TLDS: Set[str] = {
    "adult", "am", "bar", "best", "bid", "biz", "bond", "buzz", "cam", "casa",
    "cf", "cfd", "click", "cloud", "club", "country", "cricket", "cyou", "date",
    "download", "fit", "fun", "ga", "gdn", "gq", "guru", "hair", "homes",
    "host", "icu", "info", "life", "link", "live", "loan", "lol", "mba", "ml",
    "mov", "move", "online", "party", "pro", "pw", "racing", "rest", "review",
    "ru", "sbs", "science", "sex", "shop", "site", "space", "store", "stream",
    "support", "surf", "tech", "tk", "top", "trade", "vip", "wang", "win",
    "work", "world", "xxx", "xyz", "zip", "faith", "monster", "su"
}

DEFAULT_CONTEXTUAL_TLDS: Set[str] = {
    # ──────────────────────────────────────────────
    # 🌍 EUROPE (well-governed ccTLDs)
    # ──────────────────────────────────────────────
    "de", "fr", "uk", "gb", "es", "nl", "se", "no", "fi", "dk", "pl", "ch", "at",
    "be", "ie", "it", "cz", "pt", "gr", "lu", "hu", "sk", "si", "hr", "lt", "lv",
    "ee", "ro", "bg", "is",
    # Rationale: EU & EFTA nations with strict registrar verification
    # and very low phishing density (per Spamhaus < 0.2%).

    # ──────────────────────────────────────────────
    # 🌎 AMERICAS
    # ──────────────────────────────────────────────
    "us", "ca", "mx", "br", "ar", "cl", "co", "pe", "uy", "cr",
    # Stable LATAM registries (.br, .cl) enforce ID or corporate checks.
    # .us and .ca operate under strong compliance frameworks.

    # ──────────────────────────────────────────────
    # 🌏 ASIA-PACIFIC
    # ──────────────────────────────────────────────
    "in", "jp", "sg", "hk", "kr", "tw", "au", "nz", "id", "my", "ph", "th", "vn",
    # Rationale: strong national registry oversight or government-backed
    # accreditation; low abuse ratios relative to free gTLDs.

    # ──────────────────────────────────────────────
    # 🌍 MIDDLE EAST & AFRICA
    # ──────────────────────────────────────────────
    "il", "sa", "ae", "qa", "om", "kw", "za", "ng", "ma", "eg",
    # UAE, Israel, and Saudi Arabia maintain controlled registry access.
    # .za, .ng, and .ma show improved anti-abuse policy since 2023.

    # ──────────────────────────────────────────────
    # 🌐 WELL-GOVERNED LEGACY gTLDs
    # ──────────────────────────────────────────────
    # "com", "net", "org", "edu", "gov", "mil", "int",
    # These are canonical high-trust spaces (ICANN/Verisign managed).

    # ──────────────────────────────────────────────
    # 💡 STABLE MODERN gTLDs (HSTS-preloaded or enterprise-managed)
    # ──────────────────────────────────────────────
    "app", "dev", "page", "bank", "insurance", "pharmacy",
    "law", "bio", "museum", "post", "aero", "pro",
    # .app / .dev / .page are HSTS-enforced by default (Google Registry).
    # .bank / .insurance / .pharmacy enforce KYC and HTTPS validation.

    # ──────────────────────────────────────────────
    # ⚙️ STABLE SMALL-REGISTRY OR NIC-MANAGED ccTLDs
    # ──────────────────────────────────────────────
    "li", "tv", "ws", "cc", "tokyo", "tr", "me", "io", "ai",
    # .tv (Tuvalu), .me (Montenegro), .io (British Indian Ocean),
    # .ai (Anguilla) are popular for startups and tech; moderate abuse,
    # but widely used legitimately — keep contextual, not suspicious.

    # ──────────────────────────────────────────────
    # 🀄 LARGE BUT MIXED-GOVERNANCE TLDs (contextual only)
    # ──────────────────────────────────────────────
    "cn", "hk", "tw", "ru",
    # These are large, mixed-governance TLDs — not inherently abusive,
    # but should be weighted only contextually (co-occurrence logic).
}


DEFAULT_URL_SHORTENERS: Set[str] = {
    "bit.ly", "bitly.com", "goo.gl", "t.co", "tinyurl.com", "ow.ly", "is.gd", "v.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "tiny.cc", "t.ly", "rb.gy", "shrtco.de",
    "s.id", "clck.ru", "bl.ink", "shorturl.at", "adf.ly", "q.gs", "short.io",
    "short.cm", "soo.gd", "lnkd.in", "x.co", "mcaf.ee", "amzn.to", "trib.al",
    "smarturl.it", "snip.ly", "snipurl.com", "lnk.bio", "lnk.to", "bio.link",
    "bio.site", "tap.bio", "linktr.ee", "beacons.ai", "campsite.bio", "hey.bio",
    "instabio.cc", "yt.be", "youtu.be", "link.medium.com", "reut.rs", "nyti.ms",
    "shorte.st", "clk.sh", "clk.im", "linkshrink.net", "bc.vc", "adcrun.ch",
    "ouo.io", "exe.io", "exey.io", "linkvertise.com", "shrinkme.io",
    "shrinkearn.com", "shortzon.com", "cutpaid.com", "cutwin.com", "shortadd.com",
    "short.pe", "shrinkurl.io", "urlcash.net", "t.me", "telegram.me", "wa.me",
    "goo.su", "urlzs.com", "linkvertise.net", "go.microsoft.com", "redirect.vk.com",
    "vk.cc", "vk.me", "msft.it", "msft.ms", "aka.ms", "shortzy.in", "shortxlink.com",
    "ez4short.com", "gtly.to", "sharee.tech", "stfly.io", "zws.im", "dlink.me",
    "hyperurl.co", "urlr.me", "clicky.me", "linkbox.to", "tiny.lt", "myurls.co",
    "shortbitly.com", "shortbit.com", "urlbitly.net", "bit.do", "short.best",
    "clickmeter.com", "urlr.in", "go.ly", "click.ru", "shortenurl.io", "taplink.at",
    "goo.by", "shortly.cc", "shortcm.li", "urlz.fr", "sk.gy"
}

DEFAULT_FILE_HOSTING: Set[str] = {
    "storage.googleapis.com", "firebasestorage.googleapis.com", "googleusercontent.com",
    "s3.amazonaws.com", "amazonaws.com", "cloudfront.net", "blob.core.windows.net",
    "azureedge.net", "1drv.ms", "sharepoint.com", "digitaloceanspaces.com",
    "linodeobjects.com", "backblazeb2.com", "wasabisys.com", "r2.cloudflarestorage.com",
    "storage.yandexcloud.net", "drive.google.com", "docs.google.com", "dropbox.com",
    "dropboxusercontent.com", "onedrive.live.com", "box.com", "boxcloud.com",
    "icloud.com", "mega.nz", "mediafire.com", "wetransfer.com", "we.tl", "terabox.com",
    "4shared.com", "raw.githubusercontent.com", "githubusercontent.com", "gist.github.com",
    "gitlab.com", "bitbucket.org", "codeberg.org", "sendspace.com", "zippyshare.com",
    "pixeldrain.com", "krakenfiles.com", "workupload.com", "uploadfiles.io",
    "upload.ee", "uploadhaven.com", "uppit.com", "dropmefiles.com", "filedropper.com",
    "gofile.io", "catbox.moe", "uguu.se", "rapidgator.net", "nitroflare.com",
    "anonfiles.com", "anonfiles.cc", "anonfiles.me", "bayfiles.com", "userscloud.com",
    "filemail.com", "sendgb.com", "filetransfer.io", "transfernow.net", "transferxl.com",
    "filehost.ws", "files.fm", "file.io", "ufile.io", "0x0.st", "bashupload.com",
    "cdn.discordapp.com", "pastebin.com", "paste.ee", "hastebin.com", "ghostbin.com",
    "pastes.io", "justpaste.it", "controlc.com", "dpaste.org", "paste2.org",
    "termbin.com", "sprunge.us", "0paste.com", "rentry.co", "telegra.ph", "imgur.com",
    "imgbb.com", "postimg.cc", "ibb.co", "imageshack.us", "transfer.sh", "temp.sh",
    "filebin.net", "filebin.ca"
}

DEFAULT_TUNNEL_HOSTS: Set[str] = {
    "translate.google.com", "translate.googleusercontent.com", "webcache.googleusercontent.com",
    "googleweblight.com", "r.jina.ai", "cc.bingj.com", "cc.bing.com", "api.allorigins.win",
    "api.allorigins.cf", "api.allorigins.xyz", "corsproxy.io", "cors-anywhere.herokuapp.com",
    "thingproxy.freeboard.io", "l.facebook.com", "l.instagram.com", "l.messenger.com",
    "out.reddit.com", "vm.tiktok.com", "lnkd.in", "t.co", "aka.ms", "msft.it",
    "web.archive.org", "archive.is", "translate.yandex.net", "translate.yandex.com"
}

HIGH_RISK_EXTENSIONS: Set[str] = {
    ".exe", ".dll", ".scr", ".msi", ".msp", ".bat", ".cmd", ".ps1", ".vbs", ".hta", ".lnk",
    ".jar", ".war", ".ear", ".dmg", ".pkg", ".app", ".deb", ".rpm", ".apk", ".run",
    ".bin", ".sh", ".tgz", ".xapk", ".ipa", ".py", ".rb", ".pl", ".php", ".js",
    ".jse", ".psm1", ".psd1", ".cab", ".msu"
}

MEDIUM_RISK_EXTENSIONS: Set[str] = {
    ".zip", ".7z", ".rar", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".xz",
    ".iso", ".img", ".vhd", ".vhdx", ".vmdk", ".ova", ".ova.gz", ".apk", ".jar",
    ".lnk", ".gpg", ".enc", ".aes", ".pgp"
}

DOC_LURE_EXTENSIONS: Set[str] = {
    ".pdf", ".doc", ".docx", ".docm", ".dot", ".dotx", ".dotm", ".xls", ".xlsx",
    ".xlsm", ".xlt", ".ppt", ".pptx", ".pptm", ".pps", ".ppsx", ".rtf", ".odt",
    ".odp", ".ods", ".htm", ".html", ".shtml", ".xhtml", ".mht", ".mhtml", ".eml",
    ".msg", ".txt", ".log", ".csv", ".url", ".webloc", ".ics", ".pages", ".key",
    ".svg", ".webp", ".png", ".jpg", ".jpeg"
}

# High-value brand tokens frequently abused in phishing/impersonation.
# Use case-insensitive matching. For tokens of length <= 3, enforce word boundaries.
POPULAR_BRANDS: Set[str] = {
    # Tech / Cloud
    "google","gmail","gworkspace","gdrive","youtube","android","chrome","googleplay",
    "apple","appleid","icloud","itunes","mac","iphone","ipad","imac",
    "microsoft","windows","office","office365","live","outlook","hotmail","skype",
    "azure","onedrive","sharepoint","teams","github","gitlab","bitbucket","dropbox","box",
    "canva","notion","asana","atlassian","jira","confluence","salesforce","hubspot",
    "zendesk","airtable","monday","zoom","slack",

    # E-commerce / Retail / Marketplaces
    "amazon","aws","alibaba","aliexpress","ebay","shopify","walmart","target",
    "flipkart","shein","temu","rakuten","mercadolibre","bestbuy","costco","carrefour",
    "ikea","lazada",

    # Payments / Fintech
    "paypal","venmo","skrill","stripe","square","adyen","revolut","wise","payoneer",
    "klarna","afterpay","affirm","gcash","paytm","phonepe","razorpay","bharatpe","payu",

    # Banking (global + regional often spoofed)
    "chase","bankofamerica","wellsfargo","citibank","citi","capitalone","discover",
    "americanexpress","amex","usbank","pnc","tdbank","hsbc","barclays","santander",
    "rbc","scotiabank","anz","commbank","westpac","icici","hdfc","axis","sbi","kotak",
    "yesbank","dbs",

    # Social / Messaging / Identity
    "facebook","meta","instagram","threads","whatsapp","messenger","tiktok","twitter",
    "snapchat","telegram","discord","reddit","pinterest","tumblr","wechat","line",
    "signal","linkedin",

    # Streaming / Gaming / Entertainment
    "netflix","hulu","disney","disneyplus","primevideo","spotify","soundcloud","twitch",
    "crunchyroll","paramount","peacock","hbomax","max","vimeo",
    "steam","steampowered","epicgames","playstation","psn","xbox","nintendo","ea",
    "ubisoft","riot","blizzard",

    # Logistics / Postal
    "fedex","ups","dhl","usps","royalmail","canadapost","australiapost",

    # Crypto / Trading
    "binance","coinbase","kraken","bitfinex","bitstamp","blockchain","trustwallet",
    "metamask","ledger","trezor","robinhood","etrade","webull","fidelity",

    # Security / Identity / IT
    "norton","mcafee","kaspersky","bitdefender","avast","avg","okta","duo","authy",
    "lastpass","1password","cloudflare","fortinet","proofpoint","zscaler","sophos",
    "crowdstrike","sentinelone","trendmicro","fireeye","barracuda","verisign",

    # Travel / Hospitality / Airlines
    "expedia","booking","airbnb","tripadvisor","trivago","agoda","makemytrip","hotels",
    "marriott","hilton","hyatt","ihg","emirates","qatarairways","lufthansa","delta",
    "united","britishairways","airfrance","ryanair","easyjet","aa",

    # Email Providers (frequently spoofed)
    "protonmail","zoho","gmx","yahoo","ymail","aol","mailru","yandex","fastmail"
}


BRAND_DOMAINS: Set[str] = {
    # --- Core Tech / Cloud Giants ---
    "google.com", "gmail.com", "google.co.in", "youtube.com", "android.com",
    "chrome.com", "apple.com", "icloud.com", "itunes.apple.com", "mac.com",
    "microsoft.com", "windows.com", "office.com", "office365.com", "live.com",
    "outlook.com", "skype.com", "onedrive.com", "azure.com", "sharepoint.com",
    "adobe.com", "acrobat.com", "canva.com", "dropbox.com", "box.com",
    "notion.so", "asana.com", "atlassian.com", "jira.com", "confluence.com",
    "salesforce.com", "hubspot.com", "zendesk.com", "monday.com", "airtable.com",

    # --- E-Commerce / Retail / Marketplaces ---
    "amazon.com", "aws.amazon.com", "amazon.in", "alibaba.com", "aliexpress.com",
    "ebay.com", "shopify.com", "flipkart.com", "shein.com", "temu.com",
    "rakuten.com", "mercadolibre.com", "walmart.com", "target.com",
    "bestbuy.com", "costco.com", "ikea.com", "carrefour.com", "lazada.com",

    # --- Banks / Financial Institutions / Payment Gateways ---
    "paypal.com", "venmo.com", "stripe.com", "squareup.com", "skrill.com",
    "revolut.com", "wise.com", "payoneer.com", "adyen.com",
    "chase.com", "bankofamerica.com", "wellsfargo.com", "citibank.com", "citi.com",
    "capitalone.com", "discover.com", "americanexpress.com", "usbank.com",
    "pnc.com", "tdbank.com", "hsbc.com", "barclays.co.uk", "santander.com",
    "icicibank.com", "hdfcbank.com", "axisbank.com", "onlinesbi.com",
    "kotak.com", "yesbank.in", "dbs.com", "rbcroyalbank.com", "scotiabank.com",
    "anz.com", "commbank.com.au", "westpac.com.au",

    # --- Social Media / Messaging / Identity ---
    "facebook.com", "meta.com", "instagram.com", "threads.net", "whatsapp.com",
    "messenger.com", "tiktok.com", "tiktokcdn.com", "twitter.com", "x.com",
    "snapchat.com", "discord.com", "telegram.org", "reddit.com",
    "pinterest.com", "tumblr.com", "wechat.com", "line.me", "signal.org",
    "slack.com", "zoom.us", "zoom.com", "linkedin.com",

    # --- Streaming / Entertainment / Gaming ---
    "netflix.com", "hulu.com", "disneyplus.com", "disney.com", "primevideo.com",
    "spotify.com", "soundcloud.com", "twitch.tv", "crunchyroll.com",
    "hbomax.com", "max.com", "paramountplus.com", "peacocktv.com", "vimeo.com",
    "steamcommunity.com", "store.steampowered.com", "epicgames.com",
    "playstation.com", "xbox.com", "nintendo.com", "ea.com", "ubisoft.com",

    # --- Logistics / Shipping / Delivery ---
    "fedex.com", "ups.com", "dhl.com", "usps.com", "royalmail.com",
    "canadapost-postescanada.ca", "australiapost.com.au", "parcel.com",

    # --- Crypto / Fintech / Trading Platforms ---
    "binance.com", "coinbase.com", "kraken.com", "bitfinex.com", "bitstamp.net",
    "blockchain.com", "trustwallet.com", "metamask.io", "ripple.com",
    "opensea.io", "ledger.com", "trezor.io", "robinhood.com",
    "etrade.com", "webull.com", "fidelity.com", "sofi.com", "revolut.com",

    # --- Travel / Hospitality / Airlines ---
    "expedia.com", "booking.com", "airbnb.com", "tripadvisor.com", "trivago.com",
    "agoda.com", "makemytrip.com", "hotels.com", "marriott.com", "hilton.com",
    "hyatt.com", "ihg.com", "emirates.com", "qatarairways.com", "lufthansa.com",
    "delta.com", "united.com", "aa.com", "ryanair.com", "easyjet.com",
    "britishairways.com", "airfrance.com",

    # --- Government / Postal / Public Sector (frequent impersonations) ---
    "irs.gov", "hmrc.gov.uk", "gov.uk", "uscis.gov", "state.gov", "aadhar.gov.in",
    "mygov.in", "revenue.gov", "treasury.gov", "cbp.gov", "ssa.gov",

    # --- Security / Identity / IT Vendors ---
    "norton.com", "mcafee.com", "kaspersky.com", "bitdefender.com",
    "avast.com", "avg.com", "okta.com", "duo.com", "authy.com", "lastpass.com",
    "1password.com", "cloudflare.com", "fortinet.com", "proofpoint.com",
    "zscaler.com", "sophos.com", "crowdstrike.com", "sentinelone.com",
    "trendmicro.com", "fireeye.com", "barracuda.com", "verisign.com",

    # --- Telecom / ISP / Utilities ---
    "verizon.com", "att.com", "tmobile.com", "vodafone.com", "jio.com",
    "airtel.in", "orange.com", "bt.com", "comcast.com", "xfinity.com",
    "rogers.com", "bell.ca", "telus.com", "ntt.com", "chinaunicom.cn",
    "spectrum.com", "sktelecom.com", "stc.com.sa",

    # --- Education / Career / SaaS / Productivity ---
    "coursera.org", "udemy.com", "edx.org", "blackboard.com", "canvaslms.com",
    "moodle.org", "zoominfo.com", "indeed.com", "linkedin.com", "glassdoor.com",
    "workday.com", "servicenow.com", "oracle.com", "sap.com",

    # --- Regional Payments / Commerce ---
    "gcash.com", "paytm.com", "phonepe.com", "razorpay.com", "bharatpe.com",
    "payu.in", "klarna.com", "afterpay.com", "affirm.com",
    "grab.com", "gojek.com", "alipay.com", "wechatpay.com",
    "mercadopago.com", "mercadopago.com.ar", "pix.com.br",
}


IMAGE_CDN_DOMAINS: Set[str] = {
    "imgur.com", "i.imgur.com", "imgbb.com", "i.ibb.co", "ibb.co", "postimg.cc",
    "i.postimg.cc", "imageshack.us", "cdn.discordapp.com", "media.discordapp.net",
    "i.redd.it", "pbs.twimg.com", "cloudinary.com", "res.cloudinary.com",
    "gyazo.com", "i.gyazo.com", "prnt.sc"
}

SUSPICIOUS_PORTS: Set[int] = {
    8080, 8081, 8082, 8083, 8084, 8085, 8088, 8000, 8001, 8008, 8010, 8443, 8880,
    8888, 9999, 1080, 3128, 9050, 8118, 1081, 1085, 4145, 4153, 6588, 6589, 6666,
    6667, 6697, 22, 2222, 23, 3389, 5900, 5985, 5986, 3306, 5432, 27017, 6379,
    9200, 11211, 1337, 1338, 1352, 4443, 4444, 5555, 6660, 7000, 7001, 8089,
    9000, 9001, 9002, 9003, 9010, 10000, 10101, 10443, 12345, 16000, 22222, 25,
    465, 587, 2525, 110, 995, 143, 993, 2082, 2083, 2095, 2096, 8890, 9090, 9443,
    9998, 10080
}

REDIRECT_PARAMS: Set[str] = {
    "redirect", "redirect_url", "redirect_uri", "redir", "url", "u", "next",
    "continue", "dest", "destination", "target", "return", "returnto", "to", "r",
    "go", "link", "out", "target_url", "forward"
}

SUSPICIOUS_KEYWORDS: Set[str] = {
    # Credential / authentication
    "login", "logon", "signin", "signon", "authenticate", "auth", "verify",
    "verification", "validate", "credential", "session", "token", "2fa", "mfa",
    "otp", "password", "passcode",
    # Account lifecycle
    "account", "myaccount", "user", "member", "customer", "profile", "portal",
    "dashboard", "access", "unlock", "recovery", "reset", "restore", "reactivate",
    "update", "confirm", "deactivate", "suspend", "suspended", "revalidate",
    "locked", "expired", "expiration",
    # Urgency
    "alert", "urgent", "immediate", "important", "notice", "warning", "limited",
    "critical", "attention", "mandatory", "required", "deadline", "final", "now",
    "soon", "instantly",
    # Finance / payment
    "bank", "banking", "payment", "invoice", "billing", "transaction", "deposit",
    "refund", "wallet", "crypto", "bitcoin", "ether", "giftcard", "reward",
    "bonus", "prize", "cash", "payout", "salary", "loan", "mortgage", "tax",
    "customs", "fine", "penalty",
    # Shipping / delivery
    "delivery", "shipment", "tracking", "parcel", "package", "logistics", "express",
    "post", "postal", "courier",
    # Security / compliance
    "secure", "security", "safe", "safety", "compliance", "policy", "breach",
    "violation", "phishing", "malware", "virus", "scan", "quarantine",
    # Misc social engineering
    "support", "helpdesk", "service", "updateinfo", "unlockaccount", "bonus",
    "lucky", "winner", "congratulations"
}

SESSION_PARAM_NAMES: Set[str] = {
    "session", "sessionid", "sid", "sessid", "jsessionid", "phpsessid", "access_token",
    "bearer", "csrf", "xsrf", "token", "id_token", "sso", "samlresponse", "oauth",
    "code", "state", "relaystate", "service", "target_link_uri", "request_uri",
    "ticket", "auth", "key"
}

LANGUAGE_TOKENS: Set[str] = {
    "en", "en-us", "en-gb", "fr", "de", "es", "it", "pt", "pt-br", "ru", "cn", "zh",
    "zh-cn", "zh-tw", "ja", "jp", "ko", "kr", "ar", "hi", "bn", "id", "ms", "th",
    "vi", "pl", "nl", "sv", "no", "fi", "da", "cs", "sk", "el", "tr", "ro", "hu"
}

# Country-code and geopolitical TLDs with elevated risk,
# recurring abuse in phishing/malware campaigns, or state-level control issues.
# These are not automatically "malicious", but should raise context flags
# when combined with phishing, tunneling, or brand-impersonation signals.

GEO_SENSITIVE_TLDS: Set[str] = {
    # ─────────────────────────────
    # 🇷🇺 Russian Federation & Satellites
    # ─────────────────────────────
    "ru",   # Russia — chronic phishing, info-ops infrastructure
    "su",   # Soviet Union legacy TLD, still active & high abuse
    "by",   # Belarus — hosting overlap with RU infra
    "kz",   # Kazakhstan — emerging RU-aligned infra
    "uz",   # Uzbekistan — permissive registrars

    # ─────────────────────────────
    # 🇨🇳 Greater China
    # ─────────────────────────────
    "cn",   # China — large volume; mix of legit + heavy abuse
    "hk",   # Hong Kong — used in proxy and link cloaking
    "tw",   # Taiwan — frequent target & occasional abuse hub
    "mo",   # Macau — small but observed in spam chains

    # ─────────────────────────────
    # 🇮🇷 / 🇸🇾 / 🇰🇵 Sanctioned or high-control regimes
    # ─────────────────────────────
    "ir",   # Iran — restricted internet, seen in phishing relay nodes
    "sy",   # Syria — limited oversight, some gov-linked ops
    "kp",   # North Korea — closed network; any public use suspicious
    "cu",   # Cuba — restricted registry; rarely legitimate
    "sd",   # Sudan — conflict-related abuse hosting

    # ─────────────────────────────
    # 🇹🇷 / 🇺🇦 / 🇧🇷 Regional hot zones (mixed)
    # ─────────────────────────────
    "tr",   # Turkey — politically active domain space; occasional malware infra
    "ua",   # Ukraine — legitimate + war-related malicious infra (contextual)
    "br",   # Brazil — large mixed registry; recurring phishing campaigns
    "vn",   # Vietnam — used in fraud, gambling spam, C2 pivots
    "id",   # Indonesia — rising abuse due to open registration
    "pk",   # Pakistan — used for ideological phishing
    "my",   # Malaysia — mid-abuse levels, gambling & spam
    "ph",   # Philippines — common in dating scams / SMS phish

    # ─────────────────────────────
    # 🌍 Emerging or semi-regulated ccTLDs (monitor)
    # ─────────────────────────────
    "la",   # Laos — commercialized; used for link shorteners
    "cc",   # Cocos Islands — often misused for offshore infra
    "ws",   # Samoa — abused for short branding & malware delivery
    "io",   # British Indian Ocean — startup & malware overlap
    "ai",   # Anguilla — AI startup trend; phishing blend risk
}


GEO_KEYWORDS: Set[str] = {
    "gov", "bank", "customs", "tax", "irs", "parcel", "post", "prefeitura", "police",
    "military", "army", "embassy", "visa", "consulate", "payment", "pay", "banking"
}

WEBAPP_PATH_KEYWORDS: Set[str] = {
    "wp-login.php", "wp-admin", "admin", "cpanel", "panel", "login", "signin",
    "user/login", "user/signin", "umbraco", "manager", "administrator", "dashboard",
    "auth", "phpmyadmin", "console", "control", "backoffice", "portal", "account",
    "secure", "myaccount"
}

MOBILE_SUBDOMAIN_KEYWORDS: Set[str] = {"m", "mobile", "amp"}

MALICIOUS_PATTERN_STRINGS: Sequence[str] = (
    r"union\s+select",
    r"cmd=",
    r"exec\(",
    r"<script",
    r"onerror=",
    r"eval\(",
    r"\.\./\.\./",
    r"drop\s+table",
    r"insert\s+into",
    r"document\.cookie\s*=",
    r"atob\(", r"btoa\(", r"base64,"
)

BASE64_BLOB_PATTERN = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{32,}={0,2}(?![A-Za-z0-9+/=])")

# ---------------------------------------------------------------------------
# NEW INTELLIGENCE SETS FOR EXPANDED PHISHING DETECTION (18 new categories)
# ---------------------------------------------------------------------------

# URL protection / rewriting services abused to wrap malicious URLs
URL_PROTECTION_DOMAINS: Set[str] = {
    "urldefense.proofpoint.com", "urldefense.com",
    "linkprotect.cudasvc.com", "protect-us.mimecast.com",
    "protect-eu.mimecast.com", "protect-au.mimecast.com",
    "protect-de.mimecast.com", "protect-za.mimecast.com",
    "safelinks.protection.outlook.com",
    "nam01.safelinks.protection.outlook.com",
    "nam02.safelinks.protection.outlook.com",
    "eur01.safelinks.protection.outlook.com",
    "eur02.safelinks.protection.outlook.com",
    "apc01.safelinks.protection.outlook.com",
    "clicktime.symantec.com", "app.clicktime.com",
    "links.protect.barracuda.com",
    "mandrillapp.com", "click.mailerlite.com",
    "u.safebrowsing.apple.com",
    "emaildefense.proofpoint.com",
}

# Dynamic DNS providers commonly used for phishing infrastructure
DDNS_PROVIDERS: Set[str] = {
    "duckdns.org", "no-ip.com", "no-ip.org", "no-ip.biz",
    "noip.com", "noip.me", "dynu.com", "dynu.net",
    "dynv6.net", "dynv6.com",
    "hopto.org", "zapto.org", "sytes.net", "ddns.net",
    "myvnc.com", "serveftp.com", "servegame.com",
    "redirectme.net", "myftp.biz", "myftp.org",
    "loseyourip.com", "portmap.host", "freedns.afraid.org",
    "afraid.org", "changeip.com", "changeip.net", "changeip.org",
    "dns.army", "dns.navy", "chickenkiller.com",
    "crabdance.com", "jumpingcrab.com", "mooo.com",
    "strangled.net", "twillightparadox.com", "us.to",
    "selfip.com", "selfip.org", "selfip.net", "selfip.biz",
    "servehttp.com", "servebeer.com", "servecounterstrike.com",
    "servepics.com", "servequake.com",
    "gotdns.ch", "gotdns.org", "gotdns.com",
    "dnsd.me", "ddnsfree.com", "ddnsking.com",
    "giize.com", "gleeze.com", "kozow.com",
    "3utilities.com", "bounceme.net", "ddns.me",
    "myddns.me", "servehalflife.com", "serveminecraft.net",
    "webhop.me",
}

# QR code generation API domains (quishing infrastructure)
QR_API_DOMAINS: Set[str] = {
    "api.qrserver.com", "chart.googleapis.com",
    "qr.io", "qrcodes.pro", "goqr.me", "qr-code-generator.com",
    "qrcode-monkey.com", "me-qr.com", "qr.new", "qrfy.com",
    "flowcode.com", "beaconstac.com", "scanova.io",
    "unitag.io", "forqrcode.com", "qrcode.tec-it.com",
}

# Cryptocurrency scam keywords in URL paths/queries
CRYPTO_SCAM_KEYWORDS: Set[str] = {
    "connect-wallet", "connectwallet", "wallet-connect", "walletconnect",
    "claim-airdrop", "claimairdrop", "free-airdrop", "freeairdrop",
    "verify-wallet", "verifywallet", "wallet-verify", "walletverify",
    "seed-phrase", "seedphrase", "recovery-phrase", "recoveryphrase",
    "private-key", "privatekey", "import-wallet", "importwallet",
    "metamask-verify", "metamaskverify", "metamask-login", "metamasklogin",
    "token-approval", "tokenapproval", "approve-token", "approvetoken",
    "claim-tokens", "claimtokens", "free-tokens", "freetokens",
    "mint-nft", "mintnft", "free-mint", "freemint",
    "stake-rewards", "stakerewards", "claim-rewards", "claimrewards",
    "swap-token", "swaptoken", "liquidity-pool", "liquiditypool",
    "defi-yield", "defiyield", "yield-farm", "yieldfarm",
    "dex-swap", "dexswap", "pancakeswap-claim", "uniswap-claim",
    "etherscan-verify", "bscscan-verify",
    "trezor-start", "ledger-start", "phantom-auth",
}

# CAPTCHA service keywords used to shield phishing pages from scanners
CAPTCHA_KEYWORDS: Set[str] = {
    "g-recaptcha", "recaptcha", "recaptcha-response",
    "hcaptcha", "h-captcha", "hcaptcha-response",
    "cf-turnstile", "turnstile", "cf-challenge",
    "challenge-platform", "challenge-form",
    "funcaptcha", "arkoselabs",
    "geetest", "geetest_challenge",
}

# Expanded CMS admin/exploit paths frequently hosting phishing kits
CMS_EXPLOIT_PATHS: Set[str] = {
    # WordPress
    "wp-login.php", "wp-admin", "wp-includes", "wp-content/uploads",
    "wp-content/plugins", "wp-content/themes", "xmlrpc.php",
    "wp-json", "wp-cron.php",
    # Joomla
    "administrator", "joomla/administrator", "components",
    # Drupal
    "user/login", "admin/login", "drupal",
    # Magento
    "admin_panel", "magento/admin", "downloader",
    # Generic CMS / Server panels
    "cpanel", "cPanel", "plesk", "directadmin",
    "phpmyadmin", "adminer.php", "webmin",
    "filemanager", "file_manager",
    # Common phishing kit paths on compromised sites
    ".well-known", "cgi-bin", "tmp/upload",
    "sites/default/files", "misc/ajax.php",
    "owa/auth", "autodiscover", "ecp/default",
    "remote/login", "vpn/index.html",
    "citrix/", "pulse/", "global-protect",
}

# Digital publishing / content creation platforms abused for phishing
PUBLISHING_PLATFORMS: Set[str] = {
    # Google ecosystem
    "sites.google.com", "docs.google.com", "forms.google.com",
    "script.google.com", "firebase.google.com",
    "web.app", "firebaseapp.com", "page.link",
    # Website builders
    "wix.com", "wixsite.com", "wixstudio.com",
    "weebly.com", "squarespace.com",
    "webflow.io", "webflow.com",
    "carrd.co", "softr.app", "glitch.me",
    "netlify.app", "netlify.com",
    "vercel.app", "vercel.com",
    "herokuapp.com", "render.com",
    "pages.dev",  # Cloudflare Pages
    "workers.dev",  # Cloudflare Workers
    # Bio / Link-in-bio
    "linktr.ee", "bio.link", "bio.site", "lnk.bio",
    "beacons.ai", "campsite.bio", "taplink.at",
    # Collaboration / Docs
    "notion.site", "notion.so",
    "coda.io", "airtable.com",
    "canva.com",
    # Survey / Form builders
    "typeform.com", "jotform.com", "surveymonkey.com",
    "formstack.com", "cognito.com", "paperform.co",
    "tally.so", "fillout.com",
    # Blog / Publishing
    "medium.com", "substack.com",
    "ghost.io", "hashnode.dev",
    "blogspot.com", "wordpress.com",
    "telegra.ph",
    # Presentation / Document sharing
    "slideshare.net", "issuu.com",
    "flipsnack.com", "calameo.com",
}

# IPFS / Decentralized hosting gateways
IPFS_GATEWAYS: Set[str] = {
    "ipfs.io", "gateway.ipfs.io", "dweb.link",
    "cloudflare-ipfs.com", "cf-ipfs.com",
    "ipfs.infura.io", "ipfs.fleek.co",
    "gateway.pinata.cloud", "ipfs.pinata.cloud",
    "ipfs.eternum.io", "ipfs.best-practice.se",
    "hardbin.com", "ipfs.runfission.com",
    "4everland.io", "w3s.link",
    "nftstorage.link", "arweave.net", "ar.io",
    "permaweb.io",
}

# Disposable / temporary email service domains
DISPOSABLE_EMAIL_DOMAINS: Set[str] = {
    "guerrillamail.com", "guerrillamail.net", "guerrillamail.org",
    "guerrillamailblock.com", "grr.la", "sharklasers.com",
    "tempmail.com", "temp-mail.org", "temp-mail.io",
    "mailinator.com", "maildrop.cc", "yopmail.com",
    "throwaway.email", "trashmail.com", "trashmail.me",
    "dispostable.com", "mailnesia.com", "minutemail.com",
    "tempr.email", "discard.email", "mailcatch.com",
    "mohmal.com", "fakeinbox.com", "emailondeck.com",
    "getnada.com", "inboxbear.com", "mailsac.com",
    "10minutemail.com", "tempail.com",
    "tempmailaddress.com", "tempinbox.com",
    "burnermail.io", "33mail.com", "anonaddy.me",
    "simplelogin.io", "duck.com", "relay.firefox.com",
}

# Urgency / time-pressure manipulation keywords
URGENCY_KEYWORDS: Set[str] = {
    "expires-in", "expiresin", "countdown", "timer",
    "limited-time", "limitedtime", "last-chance", "lastchance",
    "final-warning", "finalwarning", "final-notice", "finalnotice",
    "hours-left", "hoursleft", "act-now", "actnow",
    "respond-immediately", "respondimmediately",
    "within-24-hours", "within24hours",
    "account-will-be-closed", "immediate-action",
    "suspend-within", "closing-your-account",
    "verify-within", "expiring-soon", "expiringsoon",
    "time-sensitive", "timesensitive",
    "urgent-action", "urgentaction",
    "claim-before", "claimbefore",
    "offer-expires", "offerexpires",
    "deadline-today", "deadlinetoday",
}

# Credential form keywords (stronger signal than general suspicious keywords)
CREDENTIAL_FORM_KEYWORDS: Set[str] = {
    "login", "log-in", "logon", "log-on",
    "signin", "sign-in", "signon", "sign-on",
    "authenticate", "auth", "authorize",
    "verify", "verification", "validate", "validation",
    "password", "passwd", "passcode",
    "2fa", "mfa", "otp", "one-time-password",
    "security-check", "securitycheck",
    "confirm-identity", "confirmidentity",
    "unlock-account", "unlockaccount",
}

# Email parameter names used for pre-filling phishing forms
EMAIL_PARAM_NAMES: Set[str] = {
    "email", "e-mail", "mail", "emailaddress", "email_address",
    "user", "username", "user_name", "userid", "user_id",
    "login", "loginid", "login_id",
    "login_hint", "loginhint",
    "account", "accountid", "account_id",
    "recipient", "to", "target_email",
    "upn", "preferred_username",
}

# Latin ↔ Cyrillic / Greek homoglyph confusable mapping
# Maps confusable Unicode characters to their ASCII Latin equivalent
CONFUSABLE_MAP: Dict[str, str] = {
    "\u0430": "a",  # Cyrillic а → Latin a
    "\u0435": "e",  # Cyrillic е → Latin e
    "\u043e": "o",  # Cyrillic о → Latin o
    "\u0440": "p",  # Cyrillic р → Latin p
    "\u0441": "c",  # Cyrillic с → Latin c
    "\u0443": "y",  # Cyrillic у → Latin y
    "\u0445": "x",  # Cyrillic х → Latin x
    "\u0456": "i",  # Cyrillic і → Latin i
    "\u0458": "j",  # Cyrillic ј → Latin j
    "\u04bb": "h",  # Cyrillic һ → Latin h
    "\u04c0": "l",  # Cyrillic Ӏ → Latin l
    "\u0501": "d",  # Cyrillic ԁ → Latin d
    "\u051b": "q",  # Cyrillic ԛ → Latin q
    "\u051d": "w",  # Cyrillic ԝ → Latin w
    "\u0261": "g",  # Latin ɡ → Latin g
    "\u0251": "a",  # Latin ɑ → Latin a
    "\u03bf": "o",  # Greek ο → Latin o
    "\u03b1": "a",  # Greek α → Latin a
    "\u03b5": "e",  # Greek ε → Latin e
    "\u03b9": "i",  # Greek ι → Latin i
    "\u03ba": "k",  # Greek κ → Latin k
    "\u03bd": "n",  # Greek ν → Latin n
    "\u03c1": "p",  # Greek ρ → Latin p
    "\u03c4": "t",  # Greek τ → Latin t
    "\u03c5": "u",  # Greek υ → Latin u
    "\u0432": "b",  # Cyrillic в → Latin b (visual)
    "\u043d": "h",  # Cyrillic н → Latin h (visual)
    "\u043c": "m",  # Cyrillic м → Latin m (visual)
    "\u0442": "t",  # Cyrillic т → Latin t (visual)
    "\u043a": "k",  # Cyrillic к → Latin k (visual)
    "\u0437": "3",  # Cyrillic з → digit 3 (visual)
    "\u0222": "8",  # Cyrillic Ȣ → digit 8 (visual)
}

# Brands mapped to their legitimate TLD set (for lookalike TLD swap detection)
BRAND_LEGITIMATE_TLDS: Dict[str, Set[str]] = {
    "google": {"com", "co.in", "co.uk", "co.jp", "com.au", "ca", "de", "fr"},
    "facebook": {"com"},
    "apple": {"com"},
    "microsoft": {"com"},
    "amazon": {"com", "in", "co.uk", "de", "fr", "co.jp", "com.au", "ca"},
    "paypal": {"com", "me"},
    "netflix": {"com"},
    "instagram": {"com"},
    "linkedin": {"com"},
    "twitter": {"com"},
    "yahoo": {"com", "co.jp", "co.in"},
    "chase": {"com"},
    "bankofamerica": {"com"},
    "wellsfargo": {"com"},
    "spotify": {"com"},
    "dropbox": {"com"},
    "adobe": {"com"},
    "zoom": {"us", "com"},
    "slack": {"com"},
    "steam": {"com"},
    "ebay": {"com", "co.uk", "de", "fr", "com.au"},
    "binance": {"com"},
    "coinbase": {"com"},
    "whatsapp": {"com"},
    "telegram": {"org"},
    "discord": {"com", "gg"},
    "reddit": {"com"},
    "tiktok": {"com"},
}


# ---------------------------------------------------------------------------
# LEVEL 1 – ORCHESTRATOR
# ---------------------------------------------------------------------------
# 1. holds every tunable knob + intelligence lists #
# 2. creates output folder + tld-extract cache  
# 3. category → severity weight (CRITICAL=5.0 … LOW=1.0)
@dataclass
class CategoryConfig:
    """Mutable configuration and intelligence tables for URL categorisation."""

    INPUT_FILE: str = "/home/hp/SHINU RATHOD/EDA/data1.csv"
    OUTPUT_DIR: Path = field(default_factory=lambda: Path("url_categories"))
    SUMMARY_REPORT: str = "categorization_report.txt"
    URL_COLUMN: str = "input"
    LABEL_COLUMN: str = "label"
    ALLOW_OVERLAPPING_CATEGORIES: bool = True
    SAVE_UNMATCHED: bool = True
    CHUNK_SIZE: int = 100_000
    WORKERS: int = 4
    DETECTION_WEIGHTS: Dict[str, float] = field(default_factory=dict)

    SUSPICIOUS_TLDS: Set[str] = field(default_factory=lambda: set(DEFAULT_SUSPICIOUS_TLDS))
    CONTEXTUAL_TLDS: Set[str] = field(default_factory=lambda: set(DEFAULT_CONTEXTUAL_TLDS))
    URL_SHORTENERS: Set[str] = field(default_factory=lambda: set(DEFAULT_URL_SHORTENERS))
    FILE_HOSTING_DOMAINS: Set[str] = field(default_factory=lambda: set(DEFAULT_FILE_HOSTING))
    TUNNEL_HOSTS: Set[str] = field(default_factory=lambda: set(DEFAULT_TUNNEL_HOSTS))
    HI_RISK_EXT: Set[str] = field(default_factory=lambda: set(HIGH_RISK_EXTENSIONS))
    MED_RISK_EXT: Set[str] = field(default_factory=lambda: set(MEDIUM_RISK_EXTENSIONS))
    DOC_LURE_EXT: Set[str] = field(default_factory=lambda: set(DOC_LURE_EXTENSIONS))
    POPULAR_BRANDS: Set[str] = field(default_factory=lambda: set(POPULAR_BRANDS))
    BRAND_DOMAINS: Set[str] = field(default_factory=lambda: set(BRAND_DOMAINS))
    IMAGE_CDN_DOMAINS: Set[str] = field(default_factory=lambda: set(IMAGE_CDN_DOMAINS))
    SUSPICIOUS_PORTS: Set[int] = field(default_factory=lambda: set(SUSPICIOUS_PORTS))
    REDIRECT_PARAMS: Set[str] = field(default_factory=lambda: set(REDIRECT_PARAMS))
    SUSPICIOUS_KEYWORDS: Set[str] = field(default_factory=lambda: set(SUSPICIOUS_KEYWORDS))
    SESSION_PARAM_NAMES: Set[str] = field(default_factory=lambda: set(SESSION_PARAM_NAMES))
    LANGUAGE_TOKENS: Set[str] = field(default_factory=lambda: set(LANGUAGE_TOKENS))
    GEO_SENSITIVE_TLDS: Set[str] = field(default_factory=lambda: set(GEO_SENSITIVE_TLDS))
    GEO_KEYWORDS: Set[str] = field(default_factory=lambda: set(GEO_KEYWORDS))
    WEBAPP_PATH_KEYWORDS: Set[str] = field(default_factory=lambda: set(WEBAPP_PATH_KEYWORDS))
    MOBILE_SUBDOMAIN_KEYWORDS: Set[str] = field(default_factory=lambda: set(MOBILE_SUBDOMAIN_KEYWORDS))

    # --- New phishing-focused intelligence sets ---
    URL_PROTECTION_SERVICES: Set[str] = field(default_factory=lambda: set(URL_PROTECTION_DOMAINS))
    DDNS_PROVIDER_DOMAINS: Set[str] = field(default_factory=lambda: set(DDNS_PROVIDERS))
    QR_API_DOMAIN_LIST: Set[str] = field(default_factory=lambda: set(QR_API_DOMAINS))
    CRYPTO_SCAM_KEYWORD_LIST: Set[str] = field(default_factory=lambda: set(CRYPTO_SCAM_KEYWORDS))
    CAPTCHA_KEYWORD_LIST: Set[str] = field(default_factory=lambda: set(CAPTCHA_KEYWORDS))
    CMS_EXPLOIT_PATH_LIST: Set[str] = field(default_factory=lambda: set(CMS_EXPLOIT_PATHS))
    PUBLISHING_PLATFORM_LIST: Set[str] = field(default_factory=lambda: set(PUBLISHING_PLATFORMS))
    IPFS_GATEWAY_LIST: Set[str] = field(default_factory=lambda: set(IPFS_GATEWAYS))
    DISPOSABLE_EMAIL_LIST: Set[str] = field(default_factory=lambda: set(DISPOSABLE_EMAIL_DOMAINS))
    URGENCY_KEYWORD_LIST: Set[str] = field(default_factory=lambda: set(URGENCY_KEYWORDS))
    CREDENTIAL_FORM_KEYWORD_LIST: Set[str] = field(default_factory=lambda: set(CREDENTIAL_FORM_KEYWORDS))
    EMAIL_PARAM_NAME_LIST: Set[str] = field(default_factory=lambda: set(EMAIL_PARAM_NAMES))
    CONFUSABLE_CHAR_MAP: Dict[str, str] = field(default_factory=lambda: dict(CONFUSABLE_MAP))
    BRAND_TLD_MAP: Dict[str, Set[str]] = field(default_factory=lambda: {k: set(v) for k, v in BRAND_LEGITIMATE_TLDS.items()})
    def __post_init__(self) -> None:
        self.WORKERS = max(1, int(self.WORKERS))
        self.CHUNK_SIZE = max(1, int(self.CHUNK_SIZE))

        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cache_dir = self.OUTPUT_DIR / ".tld_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.tld_extract = tldextract.TLDExtract(cache_dir=str(cache_dir))

        if not self.DETECTION_WEIGHTS:
            severity_weights = {"CRITICAL": 5.0, "HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}
            self.DETECTION_WEIGHTS = {
                category: severity_weights.get(info.get("severity", "MEDIUM"), 1.0)
                for category, info in URLCategory.CATEGORIES.items()}
            

        log_file = self.OUTPUT_DIR / "categorization.log"
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(log_file, encoding="utf-8", mode="w"), logging.StreamHandler()], force=True,)


# ---------------------------------------------------------------------------
# 1.2 Dataset categoriser handling chunked processing, scoring, and exports
# ---------------------------------------------------------------------------
# 1. reads CSV in 100 k chunks
# 2. ThreadPool maps URL → URLAnalyze
# 3. writes 1 csv per triggered category + cleaned csv
class URLCategorizer:
    """Processes input CSV, applies detectors, and writes outputs."""

    def __init__(self, config: CategoryConfig):
        self.config = config
        self.analyzer = URLAnalyzer(config)
        self.category_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.category_label_stats: Dict[str, Counter[str]] = defaultdict(Counter)
        self.unmatched_rows: List[Dict[str, Any]] = []
        self.all_rows: List[Dict[str, Any]] = []
        self.benign_noise_rows: List[Dict[str, Any]] = []
        self.cleaned_rows: List[Dict[str, Any]] = []
        self.explain_counter: Counter[str] = Counter()
        self.total_urls: int = 0

    def categorize_dataset(self) -> None:
        start_time = datetime.utcnow()
        logging.info("Starting categorisation pipeline")
        try:
            chunk_iter = pd.read_csv(
                self.config.INPUT_FILE,
                dtype=str,
                chunksize=self.config.CHUNK_SIZE,
                keep_default_na=False,
                na_filter=False,
                on_bad_lines="skip",
            )
        except FileNotFoundError as exc:
            logging.error("Input file not found: %s", exc)
            raise

        for chunk_index, chunk in enumerate(chunk_iter, start=1):
            logging.info("Processing chunk %s with %s rows", chunk_index, len(chunk))
            self._process_chunk(chunk)

        self._write_outputs(start_time)
        self._print_summary()

    def _process_chunk(self, chunk: pd.DataFrame) -> None:
        rows = [row._asdict() for row in chunk.itertuples(index=False, name="Row")]
        urls = [row.get(self.config.URL_COLUMN, "") for row in rows]

        if self.config.WORKERS == 1:
            analyses = [self.analyzer.analyze_url_detailed(url) for url in urls]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.WORKERS) as executor:
                analyses = list(executor.map(self.analyzer.analyze_url_detailed, urls))

        for row, analysis in zip(rows, analyses):
            url_value = row.get(self.config.URL_COLUMN, "")
            label_value = row.get(self.config.LABEL_COLUMN, "")
            matched_categories = [cat for cat, flag in analysis.flags.items() if flag]
            row_result = dict(row)
            row_result["detection_score"] = analysis.detection_score
            row_result["detection_explain"] = analysis.detection_explain
            for category, flag in analysis.flags.items():
                row_result[category] = flag

            self.all_rows.append(row_result)
            self.total_urls += 1

            if analysis.detection_explain:
                self.explain_counter[analysis.detection_explain] += 1

            if not matched_categories:
                self.unmatched_rows.append(row_result)
                self.cleaned_rows.append(row_result)
                continue

            is_benign = self._is_benign_label(label_value)
            if not (is_benign and matched_categories):
                self.cleaned_rows.append(row_result)

            for category in matched_categories:
                outcome_context = analysis.contexts.get(category)
                score_component = analysis.score_components.get(
                    category, self.config.DETECTION_WEIGHTS.get(category, 1.0)
                )
                category_entry = {
                    "url": url_value,
                    "label": label_value,
                    "matched_category": category,
                    "detection_context": outcome_context,
                    "detection_score_component": score_component,
                    "detection_score": analysis.detection_score,
                    "detection_explain": analysis.detection_explain,
                }
                self.category_rows[category].append(category_entry)

                label_key = self._label_bucket(label_value)
                self.category_label_stats[category][label_key] += 1

            if is_benign:
                benign_entry = dict(row_result)
                benign_entry["matched_categories"] = ";".join(matched_categories)
                self.benign_noise_rows.append(benign_entry)

    def _is_benign_label(self, label: Any) -> bool:
        if label is None:
            return False
        label_str = str(label).strip().lower()
        if not label_str:
            return False
        if label_str in {"0", "benign", "legit", "legitimate", "false", "negative", "good", "safe"}:
            return True
        try:
            return float(label_str) <= 0
        except ValueError:
            return False

    def _label_bucket(self, label: Any) -> str:
        if label is None or label == "":
            return "unknown"
        if self._is_benign_label(label):
            return "benign"
        label_str = str(label).strip().lower()
        if label_str in {"1", "malicious", "true", "positive", "bad"}:
            return "malicious"
        return "unknown"

    def _write_outputs(self, start_time: datetime) -> None:
        output_dir = self.config.OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV parameters to handle special characters in URLs (quotes, newlines, etc.)
        csv_params = {"index": False, "encoding": "utf-8", "quoting": csv.QUOTE_ALL, "escapechar": "\\", "doublequote": True}

        for category, rows in self.category_rows.items():
            if not rows:
                continue
            folder = output_dir / URLCategory.folder_name(category)
            folder.mkdir(parents=True, exist_ok=True)
            df_category = pd.DataFrame(rows)
            df_category.to_csv(folder / f"{category}.csv", **csv_params)

            benign_df = df_category[df_category["label"].apply(self._is_benign_label)]
            if not benign_df.empty:
                benign_df.to_csv(folder / f"benign_noise_{category}.csv", **csv_params)

        if self.benign_noise_rows:
            pd.DataFrame(self.benign_noise_rows).to_csv(output_dir / "benign_all_noise.csv", **csv_params)

        if self.cleaned_rows:
            pd.DataFrame(self.cleaned_rows).to_csv(output_dir / "data_cleaned.csv", **csv_params)

        if self.config.SAVE_UNMATCHED and self.unmatched_rows:
            unmatched_dir = output_dir / "UNMATCHED"
            unmatched_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.unmatched_rows).to_csv(unmatched_dir / "unmatched_urls.csv", **csv_params)

        self._write_summary_report(start_time)

    def _write_summary_report(self, start_time: datetime) -> None:
        report_path = self.config.OUTPUT_DIR / self.config.SUMMARY_REPORT
        duration = datetime.utcnow() - start_time

        with report_path.open("w", encoding="utf-8") as handle:
            handle.write("URL Categorization Summary\n")
            handle.write("=" * 80 + "\n\n")
            handle.write(f"Input file     : {self.config.INPUT_FILE}\n")
            handle.write(f"Processed URLs : {self.total_urls}\n")
            handle.write(f"Duration       : {duration}\n\n")

            handle.write("Per-Category Statistics\n")
            handle.write("-" * 80 + "\n")
            for category in URLCategory.categories():
                stats = self.category_label_stats.get(category)
                if not stats:
                    continue
                handle.write(f"{category}\n")
                handle.write(f"  Total      : {sum(stats.values())}\n")
                handle.write(f"  Benign     : {stats['benign']}\n")
                handle.write(f"  Malicious  : {stats['malicious']}\n")
                handle.write(f"  Unknown    : {stats['unknown']}\n\n")

            handle.write("Top Detection Explanations\n")
            handle.write("-" * 80 + "\n")
            for explanation, count in self.explain_counter.most_common(10):
                handle.write(f"{count:>6} x {explanation}\n")

    def _print_summary(self) -> None:
        logging.info("=" * 80)
        logging.info("Categorisation complete")
        for category in URLCategory.categories():
            count = len(self.category_rows.get(category, []))
            if count:
                logging.info("%s -> %s URLs", category, count)
        logging.info("-" * 80)
        logging.info("Top explanations:")
        for explanation, count in self.explain_counter.most_common(10):
            logging.info("%s x %s", count, explanation)


# ---------------------------------------------------------------------------
# LEVEL 2 – SINGLE-URL ANALYTICS ENGINE
# ---------------------------------------------------------------------------
class URLAnalyzer:
    """Applies all spreadsheet-backed detectors to a single URL."""

    _CONTEXTUAL_TLD_GATES = {
        "Shortened_URL",
        "Redirect_URL_Open_Redirect",
        "IsSuspiciousKeyword",
        "Tunneling_URLs_Proxy_Abuse",
        "IsSuspiciousFileType",
    }

    def __init__(self, config: CategoryConfig):
        self.config = config
        self._punycode_pattern = re.compile(r"(^|\.)xn--", re.IGNORECASE)
        self._unicode_pattern = re.compile(r"[^\x00-\x7F]")
        self._hex_pattern = re.compile(r"%[0-9A-Fa-f]{2}")
        self._excessive_hex_pattern = re.compile(r"(?:%[0-9A-Fa-f]{2}){5,}")
        self._decimal_ip_pattern = re.compile(r"^\d{8,10}$")
        self._hex_ip_pattern = re.compile(r"^(?:0x)?[0-9A-Fa-f]{8}$")
        self._scheme_pattern = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
        self._windows_drive_pattern = re.compile(r"^[a-zA-Z]:\\\\")
        self._repeat_subdomain_pattern = re.compile(r"(^|\.)([a-z0-9-]+)(?:\.\2){1,}\.", re.IGNORECASE)
        self._amp_path_pattern = re.compile(r"/(?:amp|amphtml)(?:/|$)", re.IGNORECASE)
        self._brand_regex_cache: Dict[str, re.Pattern[str]] = {}
        self._malicious_patterns = [re.compile(pat, re.IGNORECASE) for pat in MALICIOUS_PATTERN_STRINGS]
        self._email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

        self.detectors: List[Tuple[str, Any]] = [
            ("TypoSquatting_URL", self._is_typosquatting),
            ("Shortened_URL", self._is_shortened),
            ("Punycode_URL", self._is_punycode),
            ("Unicode_URL", self._is_unicode),
            ("Hex_Encoded_URL", self._is_hex_encoded),
            ("IP_Address_Unusual_Port_URL", self._is_ip_or_unusual_port),
            ("Decimal_Hex_IP_URL", self._is_decimal_hex_ip),
            ("Data_URL", self._is_data_url),
            ("JavaScript_URL", self._is_javascript_url),
            ("File_URL", self._is_file_url),
            ("FTP_SFTP_URL", self._is_ftp_sftp),
            ("Blob_URL", self._is_blob_url),
            ("Anchor_Fragment_Based_URL", self._is_anchor_fragment_targeted),
            ("Redirect_URL_Open_Redirect", self._has_open_redirect),
            ("Suspiciously_Long_Complex_URL", self._is_suspiciously_long_or_complex),
            ("Fake_Subdomain_URL", self._is_fake_subdomain),
            ("Chrome_Internal_URL", self._is_chrome_internal),
            ("Suspicious_TLD_URL", self._is_suspicious_tld),
            ("Tunneling_URLs_Proxy_Abuse", self._is_tunneling_proxy_abuse),
            ("Mobile_AMP_URLs", self._is_mobile_amp),
            ("HasExcessiveParams", self._has_excessive_params),
            ("HasRepeatedSubdomain", self._has_repeated_subdomain),
            ("IsBrandImpersonation", self._is_brand_impersonation),
            ("IsDynamicQuery", self._is_dynamic_query),
            ("IsGeoLocationSpecific", self._is_geolocation_specific),
            ("IsLanguageSpecific", self._is_language_specific),
            ("IsObfuscatedURL", self._is_obfuscated_url),
            ("IsSessionBased", self._is_session_token_exposed),
            ("IsSuspiciousFileType", self._is_suspicious_filetype),
            ("IsSuspiciousKeyword", self._has_suspicious_keyword),
            ("IsMaliciousPattern", self._has_malicious_pattern),
            ("IsWebAppPath", self._is_webapp_path),
            ("Structural_Malformation_URL", self._is_structural_malformation),
            ("Very_Short_URL", self._is_very_short),
            ("Non_Alpha_Start_URL", self._has_non_alpha_start_host),
            ("Cloud_Hosting_Abuse_URL", self._is_cloud_hosting_abuse),
            # --- 18 New Phishing-Focused Detectors ---
            ("Credential_Harvesting_Form_URL", self._is_credential_harvesting_form),
            ("Multi_Redirect_Chain_URL", self._has_multi_redirect_chain),
            ("URL_Protection_Service_Abuse", self._is_url_protection_abuse),
            ("Homoglyph_Domain_URL", self._is_homoglyph_domain),
            ("DNS_Wildcard_Infinite_Subdomain", self._is_dns_wildcard_subdomain),
            ("CAPTCHA_Shield_URL", self._is_captcha_shield),
            ("Compromised_CMS_URL", self._is_compromised_cms),
            ("QR_Code_Phishing_URL", self._is_qr_code_phishing),
            ("Cryptocurrency_Scam_URL", self._is_crypto_scam),
            ("Dynamic_DNS_URL", self._is_dynamic_dns),
            ("Embedded_Email_Target_URL", self._has_embedded_email_target),
            ("Subdomain_Depth_Abuse_URL", self._has_subdomain_depth_abuse),
            ("Digital_Publishing_Platform_Abuse", self._is_publishing_platform_abuse),
            ("Lookalike_TLD_Swap_URL", self._is_lookalike_tld_swap),
            ("Disposable_Email_Abuse_URL", self._is_disposable_email_abuse),
            ("Social_Engineering_Urgency_URL", self._has_urgency_manipulation),
            ("IPFS_Decentralized_Hosting_URL", self._is_ipfs_hosting),
            ("Telegram_Bot_URL", self._is_telegram_bot),
            ("Non_English_Characters_URL", self._has_non_english_chars),
            # --- Refinement v6.3: Specialized Phishing Evasion Detectors ---
            ("Nested_Encoding_Bypass_URL", self._is_nested_encoding_bypass),
            ("Deep_Domain_Stacking_URL", self._is_deep_domain_stacking),
            ("Phishing_Bypass_Path_URL", self._is_phishing_bypass_path),
            ("Public_Document_Abuse_URL", self._is_public_document_abuse),
            ("Protocol_Keyword_Subdomain_URL", self._is_protocol_keyword_subdomain),
        ]

    def analyze_url(self, url: str) -> Dict[str, bool]:
        """Maintains backward compatibility by returning category -> bool map."""
        return self.analyze_url_detailed(url).flags

    def analyze_url_detailed(self, url: str) -> AnalysisDetails:
        """Runs all detectors and returns explainable scoring details."""
        context = self._build_context(url)
        outcomes: Dict[str, DetectorOutcome] = {}

        for category, detector in self.detectors:
            try:
                outcomes[category] = detector(context)
            except Exception as exc:  # pragma: no cover - defensive logging
                logging.debug("Detector %s failed for URL %s: %s", category, context.normalized_url, exc)
                outcomes[category] = DetectorOutcome(False)

        self._apply_contextual_rules(context, outcomes)

        flags = {category: outcome.triggered for category, outcome in outcomes.items()}
        contexts = {
            category: outcome.context
            for category, outcome in outcomes.items()
            if outcome.triggered and outcome.context
        }
        score_components: Dict[str, float] = {}
        for category, outcome in outcomes.items():
            if outcome.triggered:
                weight = outcome.score_component
                if weight is None:
                    weight = self.config.DETECTION_WEIGHTS.get(category, 1.0)
                score_components[category] = weight
        
        detection_score = sum(score_components.values())
        active_categories = [category for category, outcome in outcomes.items() if outcome.triggered]

        explain_parts: List[str] = []
        for category in active_categories:
            friendly = URLCategory.friendly_name(category)
            context_str = contexts.get(category)
            explain_parts.append(f"{friendly}:{context_str}" if context_str else friendly)
        detection_explain = "; ".join(explain_parts)

        return AnalysisDetails(
            flags=flags,
            contexts=contexts,
            score_components=score_components,
            detection_score=detection_score,
            detection_explain=detection_explain,
            active_categories=active_categories,
        )

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------
    def _normalize_url(self, url: str) -> str:
        candidate = (url or "").strip()
        if not candidate:
            return "http://"
        if self._scheme_pattern.match(candidate):
            return candidate
        if candidate.startswith(("javascript:", "data:", "file:", "blob:", "chrome:", "about:", "edge:", "ftp:", "sftp:")):
            return candidate
        if candidate.startswith("\\\\") or self._windows_drive_pattern.match(candidate.replace("/", "\\")):
            return f"file:///{candidate.lstrip('/')}"
        return f"http://{candidate}"

    def _build_context(self, url: str) -> URLContext:
        normalized = self._normalize_url(url)
        try:
            parsed = urlparse(normalized)
        except ValueError as ve:
            message = str(ve)
            logging.warning("[StructuralMalformation] Parser failure (%s): %s", message, normalized)
            return URLContext(
                raw_url=url,
                normalized_url=normalized,
                scheme="",
                host="",
                port=None,
                path="",
                query="",
                fragment="",
                subdomain="",
                domain="",
                suffix="",
                registrable_domain="",
                full_domain="",
                query_params={},
                path_segments=[],
                url_length=len(normalized),
                path_depth=0,
                num_params=0,
                entropy=0.0,
                digit_ratio=0.0,
                structural_malformed=True,
                parse_error=message,
            )
        scheme = (parsed.scheme or "").lower()
        try:
            host = (parsed.hostname or "").lower()
        except ValueError as ve:
            message = str(ve)
            logging.warning("[StructuralMalformation] Host parse failure (%s): %s", message, normalized)
            return URLContext(
                raw_url=url,
                normalized_url=normalized,
                scheme=scheme,
                host="",
                port=None,
                path="",
                query="",
                fragment="",
                subdomain="",
                domain="",
                suffix="",
                registrable_domain="",
                full_domain="",
                query_params={},
                path_segments=[],
                url_length=len(normalized),
                path_depth=0,
                num_params=0,
                entropy=0.0,
                digit_ratio=0.0,
                structural_malformed=True,
                parse_error=message,
            )

        try:
            port = parsed.port
        except ValueError as ve:
            message = str(ve)
            logging.warning("[StructuralMalformation] Port parse failure (%s): %s", message, normalized)
            return URLContext(
                raw_url=url,
                normalized_url=normalized,
                scheme=scheme,
                host=host,
                port=None,
                path="",
                query="",
                fragment="",
                subdomain="",
                domain="",
                suffix="",
                registrable_domain="",
                full_domain="",
                query_params={},
                path_segments=[],
                url_length=len(normalized),
                path_depth=0,
                num_params=0,
                entropy=0.0,
                digit_ratio=0.0,
                structural_malformed=True,
                parse_error=message,
            )
        path = parsed.path or ""
        query = parsed.query or ""
        fragment = parsed.fragment or ""

        extracted = self.config.tld_extract(normalized)
        subdomain = extracted.subdomain or ""
        domain = extracted.domain or ""
        suffix = extracted.suffix or ""
        registrable = f"{domain}.{suffix}" if domain and suffix else domain or suffix or host
        full_domain = f"{domain}.{suffix}" if domain and suffix else host

        params = defaultdict(list)
        for key, value in parse_qsl(query, keep_blank_values=True):
            params[key.lower()].append(value)

        path_segments = [segment for segment in path.split("/") if segment]
        url_length = len(normalized)
        path_depth = len(path_segments)
        num_params = sum(len(values) for values in params.values())
        entropy = self._shannon_entropy(f"{host}{path}{query}")
        digit_ratio = self._digit_ratio(f"{host}{path}{query}")

        return URLContext(
            raw_url=url,
            normalized_url=normalized,
            scheme=scheme,
            host=host,
            port=port,
            path=path,
            query=query,
            fragment=fragment,
            subdomain=subdomain,
            domain=domain,
            suffix=suffix.lower(),
            registrable_domain=registrable.lower(),
            full_domain=full_domain,
            query_params=dict(params),
            path_segments=path_segments,
            url_length=url_length,
            path_depth=path_depth,
            num_params=num_params,
            entropy=entropy,
            digit_ratio=digit_ratio,
            structural_malformed=False,
            parse_error=None,
        )

    def _apply_contextual_rules(self, context: URLContext, outcomes: Dict[str, DetectorOutcome]) -> None:
        """Enforces spreadsheet gating, e.g., contextual TLDs require co-signal."""
        outcome = outcomes.get("Suspicious_TLD_URL")
        if outcome and outcome.triggered and context.suffix in self.config.CONTEXTUAL_TLDS:
            if not any(outcomes.get(cat, DetectorOutcome(False)).triggered for cat in self._CONTEXTUAL_TLD_GATES):
                outcome.triggered = False
                outcome.context = None
                outcome.score_component = None        

    @staticmethod
    def _shannon_entropy(value: str) -> float:
        if not value:
            return 0.0
        counts = Counter(value)
        length = len(value)
        return -sum((count / length) * math.log2(count / length) for count in counts.values())

    @staticmethod
    def _digit_ratio(value: str) -> float:
        if not value:
            return 0.0
        digits = sum(ch.isdigit() for ch in value)
        alphanum = sum(ch.isalnum() for ch in value)
        if alphanum == 0:
            return 0.0
        return digits / alphanum


    # ------------------------------------------------------------------
    # Detector implementations (Rows 1-33)
    # ------------------------------------------------------------------
    def _is_structural_malformation(self, context: URLContext) -> DetectorOutcome:
        """
        Detect parser or grammar-level malformations (e.g., invalid IPv6 literals,
        missing schemes with reserved delimiters, or unbalanced brackets).
        Whenever urllib.parse.urlparse() fails due to:
        ValueError: Invalid IPv6 URL
        ValueError: Invalid URL
        ValueError: unknown url type
        """
        if context.structural_malformed:
            reason = context.parse_error or "parser_error"
            return DetectorOutcome(True, context=reason)

        candidate = context.normalized_url.lower()
        if not candidate:
            return DetectorOutcome(False)

        if candidate.startswith(("file://[::", "http://[::", "https://[::")):
            return DetectorOutcome(True, context="malformed_ipv6_literal")

        if candidate.count("[") != candidate.count("]"):
            return DetectorOutcome(True, context="unbalanced_brackets")

        if "://" not in candidate and any(ch in candidate for ch in (":", "\\")):
            return DetectorOutcome(True, context="missing_scheme")

        if len(candidate) > 120 and not re.fullmatch(r"[A-Za-z0-9%:/._-]+", candidate):
            return DetectorOutcome(True, context="opaque_payload")

        return DetectorOutcome(False)

    def _is_typosquatting(self, context: URLContext) -> DetectorOutcome:
        """Row 1: Detect domains within small edit distance of popular brands."""
        domain = context.domain.lower()
        if not domain or context.full_domain in self.config.BRAND_DOMAINS:
            return DetectorOutcome(False)

        best_distance = None
        best_brand = None
        for brand in self.config.POPULAR_BRANDS:
            if not brand or brand[0] != domain[:1]:
                continue
            threshold = 1 if len(domain) <= 5 else 2
            distance = self._levenshtein(domain, brand)
            if distance == 0 or distance > threshold:
                continue
            best_distance = distance
            best_brand = brand
            break

        if best_brand is None:
            return DetectorOutcome(False)

        context_str = f"{best_brand}->{domain};lev={best_distance}"
        return DetectorOutcome(True, context=context_str)

    def _is_shortened(self, context: URLContext) -> DetectorOutcome:
        """Row 2: Detect known URL shortener domains."""
        registrable = context.registrable_domain
        if registrable in self.config.URL_SHORTENERS:
            return DetectorOutcome(True, context=registrable)
        return DetectorOutcome(False)

    def _is_punycode(self, context: URLContext) -> DetectorOutcome:
        """Row 3: Detect xn-- punycode hostnames."""
        if not context.host or not self._punycode_pattern.search(context.host):
            return DetectorOutcome(False)
        try:
            decoded = context.host.encode("ascii").decode("idna")
        except Exception:
            decoded = "<decode-error>"
        return DetectorOutcome(True, context=decoded)

    def _is_unicode(self, context: URLContext) -> DetectorOutcome:
        """Row 4: Non-ASCII characters in host or path."""
        haystack = context.host + context.path
        if self._unicode_pattern.search(haystack):
            return DetectorOutcome(True, context="non_ascii")
        return DetectorOutcome(False)

    def _is_hex_encoded(self, context: URLContext) -> DetectorOutcome:
        """Row 5: Excessive percent-encoding or encoded payloads."""
        matches = self._hex_pattern.findall(context.normalized_url)
        if len(matches) >= 5 or self._excessive_hex_pattern.search(context.normalized_url):
            return DetectorOutcome(True, context=f"percent_encoded_count={len(matches)}")

        try:
            decoded = unquote(context.normalized_url)
        except Exception:
            decoded = context.normalized_url
        if decoded.lower().startswith(("javascript:", "data:", "vbscript:", "file:")):
            return DetectorOutcome(True, context="decoded_scheme")
        return DetectorOutcome(False)

    def _is_ip_or_unusual_port(self, context: URLContext) -> DetectorOutcome:
        """Row 6: Direct IP usage or suspicious port."""
        context_info: List[str] = []
        host = context.host
        if host:
            try:
                ipaddress.ip_address(host)
                context_info.append("ip_host")
            except ValueError:
                pass
        if context.port and context.port in self.config.SUSPICIOUS_PORTS:
            context_info.append(f"port={context.port}")
        if context_info:
            context_str = ",".join(context_info)
            return DetectorOutcome(True, context=context_str)
        return DetectorOutcome(False)

    def _is_decimal_hex_ip(self, context: URLContext) -> DetectorOutcome:
        """Row 7: Hex or decimal encoded IPv4 hosts."""
        host = context.host
        if not host:
            return DetectorOutcome(False)
        host_no_dot = host.replace(".", "")
        try:
            if self._decimal_ip_pattern.fullmatch(host_no_dot):
                value = int(host_no_dot)
                ip = ipaddress.IPv4Address(value)
                return DetectorOutcome(True, context=f"decimal->{ip}")
            if self._hex_ip_pattern.fullmatch(host_no_dot):
                value = int(host_no_dot, 16)
                ip = ipaddress.IPv4Address(value)
                return DetectorOutcome(True, context=f"hex->{ip}")
        except Exception:
            logging.debug("Decimal/hex IP decode failed for %s", host)
        return DetectorOutcome(False)

    def _is_data_url(self, context: URLContext) -> DetectorOutcome:
        """Row 8: data: scheme injection."""
        return DetectorOutcome(context.scheme == "data", context=context.scheme if context.scheme == "data" else None)

    def _is_javascript_url(self, context: URLContext) -> DetectorOutcome:
        """Row 9: javascript: scheme execution."""
        return DetectorOutcome(context.scheme == "javascript", context="javascript")

    def _is_file_url(self, context: URLContext) -> DetectorOutcome:
        """Row 10: file:// or local drive paths."""
        if context.scheme == "file":
            return DetectorOutcome(True, context="file_scheme")
        if context.raw_url and (context.raw_url.startswith("\\\\") or self._windows_drive_pattern.match(context.raw_url.replace("/", "\\"))):
            return DetectorOutcome(True, context="windows_path")
        return DetectorOutcome(False)

    def _is_ftp_sftp(self, context: URLContext) -> DetectorOutcome:
        """Row 11: FTP/SFTP schemes."""
        if context.scheme in {"ftp", "sftp"}:
            return DetectorOutcome(True, context=context.scheme)
        return DetectorOutcome(False)

    def _is_blob_url(self, context: URLContext) -> DetectorOutcome:
        """Row 12: blob: URLs (often used to deliver scripts)."""
        return DetectorOutcome(context.scheme == "blob", context="blob" if context.scheme == "blob" else None)

    def _is_anchor_fragment_targeted(self, context: URLContext) -> DetectorOutcome:
        """Row 13: Fragments that contain URLs."""
        fragment = context.fragment.lower()
        if fragment and ("http://" in fragment or "https://" in fragment or "//" in fragment):
            return DetectorOutcome(True, context="fragment_contains_url")
        return DetectorOutcome(False)

    def _has_open_redirect(self, context: URLContext) -> DetectorOutcome:
        """Row 14: Query parameters pointing to an external URL."""
        for key, values in context.query_params.items():
            if key not in self.config.REDIRECT_PARAMS and "redirect" not in key and "url" not in key:
                continue
            for value in values:
                if self._value_looks_like_url(value):
                    return DetectorOutcome(True, context=f"{key}={value}")
        return DetectorOutcome(False)

    def _is_suspiciously_long_or_complex(self, context: URLContext) -> DetectorOutcome:
        """Row 15: Very long URLs or deep path structures."""
        reasons: List[str] = []
        if context.url_length > 450:
            reasons.append(f"len={context.url_length}")
        if context.path_depth > 8:
            reasons.append(f"depth={context.path_depth}")
        if context.num_params > 10:
            reasons.append(f"params={context.num_params}")
        if context.entropy > 4.5:
            reasons.append(f"entropy={context.entropy:.2f}")
        if reasons:
            return DetectorOutcome(True, context=",".join(reasons))
        return DetectorOutcome(False)

    def _is_fake_subdomain(self, context: URLContext) -> DetectorOutcome:
        """Row 16: Brand-token in subdomain masking another registrable domain."""
        if not context.subdomain:
            return DetectorOutcome(False)
        hits = self._brand_hits(context.subdomain)
        for brand in hits:
            if context.registrable_domain not in self.config.BRAND_DOMAINS:
                return DetectorOutcome(True, context=f"{brand}@{context.registrable_domain}")
        return DetectorOutcome(False)

    def _is_chrome_internal(self, context: URLContext) -> DetectorOutcome:
        """Row 17: Schemes mimicking browser internal pages."""
        if context.scheme in {"chrome", "edge", "about", "moz-extension", "chrome-extension"}:
            return DetectorOutcome(True, context=context.scheme)
        return DetectorOutcome(False)

    def _is_suspicious_tld(self, context: URLContext) -> DetectorOutcome:
        """Row 18: Suspicious or contextual TLD usage."""
        suffix = context.suffix.lower()
        if suffix in self.config.SUSPICIOUS_TLDS or suffix in self.config.CONTEXTUAL_TLDS:
            return DetectorOutcome(True, context=suffix)
        return DetectorOutcome(False)

    def _is_tunneling_proxy_abuse(self, context: URLContext) -> DetectorOutcome:
        """Row 19: Known proxy/translate/cache hosts with embedded URLs."""
        host = context.host
        if not host:
            return DetectorOutcome(False)
        if host in self.config.TUNNEL_HOSTS:
            return DetectorOutcome(True, context=host)
        if any(keyword in host for keyword in ("translate", "proxy", "cache")) and ("http://" in context.path or "http://" in context.query):
            return DetectorOutcome(True, context="proxy_keyword")
        return DetectorOutcome(False)

    def _is_cloud_hosting_abuse(self, context: URLContext) -> DetectorOutcome:
        """
        Detect URLs hosted on cloud storage/file hosting platforms often abused for phishing.
        
        Cloud hosting services like AWS S3, Google Cloud Storage, Azure Blob, Discord CDN,
        and file sharing platforms (Dropbox, MEGA, etc.) are commonly abused to host:
        - Phishing landing pages
        - Malware payloads
        - Credential harvesting forms
        
        Returns:
            DetectorOutcome with matched domain if hosted on known cloud platform.
        """
        host = context.host
        if not host:
            return DetectorOutcome(False)
        
        # Check direct match
        if host in self.config.FILE_HOSTING_DOMAINS:
            return DetectorOutcome(True, context=host)
        
        # Check if host ends with a known file hosting domain (subdomain support)
        for cloud_host in self.config.FILE_HOSTING_DOMAINS:
            if host.endswith(f".{cloud_host}") or host == cloud_host:
                return DetectorOutcome(True, context=cloud_host)
        
        # Additional cloud-specific patterns
        cloud_patterns = [
            "s3.amazonaws", "storage.googleapis", "blob.core.windows",
            "cloudfront.net", "firebasestorage", "digitaloceanspaces",
            "r2.cloudflarestorage", "cdn.discordapp"
        ]
        for pattern in cloud_patterns:
            if pattern in host:
                return DetectorOutcome(True, context=f"pattern:{pattern}")
        
        return DetectorOutcome(False)

    def _is_mobile_amp(self, context: URLContext) -> DetectorOutcome:
        """Row 20: AMP caches or mobile subdomains."""
        if context.subdomain in self.config.MOBILE_SUBDOMAIN_KEYWORDS:
            return DetectorOutcome(True, context="mobile_subdomain")
        if self._amp_path_pattern.search(context.path):
            return DetectorOutcome(True, context="amp_path")
        if "ampproject" in context.host:
            return DetectorOutcome(True, context="amp_cache")
        return DetectorOutcome(False)

    def _has_excessive_params(self, context: URLContext) -> DetectorOutcome:
        """Row 21: More than six query parameters."""
        if context.num_params > 6:
            return DetectorOutcome(True, context=f"params={context.num_params}")
        return DetectorOutcome(False)

    def _has_repeated_subdomain(self, context: URLContext) -> DetectorOutcome:
        """Row 22: Repeated subdomain labels such as login.login.example.com."""
        if not context.subdomain:
            return DetectorOutcome(False)
        if self._repeat_subdomain_pattern.search(f"{context.subdomain}."):
            return DetectorOutcome(True, context=context.subdomain)
        return DetectorOutcome(False)

    def _is_brand_impersonation(self, context: URLContext) -> DetectorOutcome:
        """Row 23: Brand keywords present but non-authoritative registrable domain."""
        haystack = f"{context.host}/{context.path}".lower()
        hits = self._brand_hits(haystack)
        for brand in hits:
            if context.registrable_domain not in self.config.BRAND_DOMAINS:
                return DetectorOutcome(True, context=f"{brand}@{context.registrable_domain}")
        return DetectorOutcome(False)

    def _is_dynamic_query(self, context: URLContext) -> DetectorOutcome:
        """Row 24: Dynamic query parameters requesting session/user identifiers."""
        for key, values in context.query_params.items():
            key_lower = key.lower()
            if any(token in key_lower for token in ("session", "token", "auth", "sid", "userid", "user")):
                for value in values:
                    if len(value) >= 12 and self._looks_random(value):
                        return DetectorOutcome(True, context=f"{key}={value[:8]}...")
        return DetectorOutcome(False)

    def _is_geolocation_specific(self, context: URLContext) -> DetectorOutcome:
        """Row 25: Geo-targeted lure using specific TLDs and keywords."""
        suffix = context.suffix.lower()
        if suffix in self.config.GEO_SENSITIVE_TLDS:
            joined = f"{context.path.lower()} {context.query.lower()}"
            if any(keyword in joined for keyword in self.config.GEO_KEYWORDS):
                return DetectorOutcome(True, context=f"{suffix}_geo_keyword")
        return DetectorOutcome(False)

    def _is_language_specific(self, context: URLContext) -> DetectorOutcome:
        """Row 26: Locale tokens present in path or query."""
        for segment in context.path_segments:
            if segment.lower() in self.config.LANGUAGE_TOKENS:
                return DetectorOutcome(True, context=f"path={segment.lower()}")
        for key, values in context.query_params.items():
            if key in {"lang", "locale", "hl", "lc"}:
                for value in values:
                    if value.lower() in self.config.LANGUAGE_TOKENS:
                        return DetectorOutcome(True, context=f"{key}={value.lower()}")
        return DetectorOutcome(False)

    def _is_obfuscated_url(self, context: URLContext) -> DetectorOutcome:
        """Row 27: High entropy or long random strings in path."""
        if context.entropy > 4.8 and len(context.path.replace("/", "")) > 40:
            return DetectorOutcome(True, context=f"entropy={context.entropy:.2f}")
        if BASE64_BLOB_PATTERN.search(context.path) or BASE64_BLOB_PATTERN.search(context.query):
            return DetectorOutcome(True, context="base64_blob")
        if re.search(r"[A-F0-9]{16,}", context.path, re.IGNORECASE):
            return DetectorOutcome(True, context="hex_blob")
        return DetectorOutcome(False)

    def _is_session_token_exposed(self, context: URLContext) -> DetectorOutcome:
        """Row 28: Session token parameters with random values."""
        for key, values in context.query_params.items():
            if key not in self.config.SESSION_PARAM_NAMES:
                continue
            for value in values:
                if len(value) >= 12 and self._looks_random(value):
                    return DetectorOutcome(True, context=f"{key}=<token>")
        return DetectorOutcome(False)

    def _is_suspicious_filetype(self, context: URLContext) -> DetectorOutcome:
        """Row 29: High-risk file extensions and double-extension traps."""
        path_lower = context.path.lower()
        if not path_lower:
            return DetectorOutcome(False)
        filename = path_lower.rsplit("/", 1)[-1]

        if filename.count(".") >= 2:
            parts = filename.split(".")
            last_ext = f".{parts[-1]}"
            prev_ext = f".{parts[-2]}"
            if last_ext in self.config.HI_RISK_EXT and prev_ext in (self.config.DOC_LURE_EXT | self.config.MED_RISK_EXT):
                registrable = context.registrable_domain
                if registrable in self.config.IMAGE_CDN_DOMAINS or context.host in self.config.IMAGE_CDN_DOMAINS:
                    weight = self.config.DETECTION_WEIGHTS.get("IsSuspiciousFileType", 1.0) * 0.5
                    return DetectorOutcome(True, context="double_ext_on_image_cdn", score_component=weight)
                return DetectorOutcome(True, context=f"double_ext={prev_ext}->{last_ext}")

        for ext in self.config.HI_RISK_EXT | self.config.MED_RISK_EXT:
            if filename.endswith(ext):
                return DetectorOutcome(True, context=ext)

        return DetectorOutcome(False)

    def _has_suspicious_keyword(self, context: URLContext) -> DetectorOutcome:
        """Row 30: Suspicious keyword presence with boundary controls."""
        haystack = f"{context.host}/{context.path}?{context.query}".lower()
        for keyword in self.config.SUSPICIOUS_KEYWORDS:
            if len(keyword) <= 4:
                pattern = self._brand_regex_cache.setdefault(
                    f"kw_{keyword}",
                    re.compile(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"),
                )
                if pattern.search(haystack):
                    return DetectorOutcome(True, context=keyword)
            else:
                pattern = self._brand_regex_cache.setdefault(
                    f"kw_full_{keyword}",
                    re.compile(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])|{re.escape(keyword)}(?=[\./\-]|$)"),
                )
                if pattern.search(haystack):
                    return DetectorOutcome(True, context=keyword)
        return DetectorOutcome(False)

    def _has_malicious_pattern(self, context: URLContext) -> DetectorOutcome:
        """Row 31: SQLi, command injection or XSS signatures."""
        haystack = f"{context.path.lower()} {context.query.lower()}"
        for pattern in self._malicious_patterns:
            if pattern.search(haystack):
                return DetectorOutcome(True, context=pattern.pattern)
        return DetectorOutcome(False)

    def _is_webapp_path(self, context: URLContext) -> DetectorOutcome:
        """Row 32: Known administrative or login webapp paths."""
        path_lower = context.path.lower()
        for keyword in self.config.WEBAPP_PATH_KEYWORDS:
            if keyword in path_lower:
                return DetectorOutcome(True, context=keyword)
        return DetectorOutcome(False)

    def _is_very_short(self, context: URLContext) -> DetectorOutcome:
        """Detect URLs that are unusually short (e.g., masking/redirection lures)."""
        if context.url_length > 0 and context.url_length < 15:
            return DetectorOutcome(True, context=f"len={context.url_length}")
        return DetectorOutcome(False)

    def _has_non_alpha_start_host(self, context: URLContext) -> DetectorOutcome:
        """Detect hostnames starting with non-alphabetic characters (digits, hex, IP, etc.)."""
        host = context.host
        if not host:
            return DetectorOutcome(False)
        
        first_char = host[0]
        if not first_char.isalpha():
            # Classify the type of non-alpha start
            if first_char.isdigit():
                reason = "digit_start"
            elif host.startswith("0x"):
                reason = "hex_start"
            elif first_char in {".", ":"}:
                reason = "delimiter_start"
            else:
                reason = "special_char_start"
            return DetectorOutcome(True, context=reason)
        return DetectorOutcome(False)

    # ------------------------------------------------------------------
    # NEW PHISHING DETECTORS (18 categories)
    # ------------------------------------------------------------------

    def _is_credential_harvesting_form(self, context: URLContext) -> DetectorOutcome:
        """Detect URLs mimicking credential forms on non-brand domains.
        Combines credential keywords + brand presence + non-authoritative domain
        for a high-confidence phishing signal."""
        if not context.host or context.registrable_domain in self.config.BRAND_DOMAINS:
            return DetectorOutcome(False)

        path_lower = context.path.lower()
        query_lower = context.query.lower()
        haystack = f"{path_lower}?{query_lower}"

        # Check for credential form keywords in path/query
        cred_hit = None
        for keyword in self.config.CREDENTIAL_FORM_KEYWORD_LIST:
            if keyword in haystack:
                cred_hit = keyword
                break
        if not cred_hit:
            return DetectorOutcome(False)

        # Check if any brand token is present in the full URL
        full_haystack = f"{context.host}/{context.path}".lower()
        brand_hits = self._brand_hits(full_haystack)
        if brand_hits:
            brand = next(iter(brand_hits))
            return DetectorOutcome(True, context=f"{cred_hit}+{brand}@{context.registrable_domain}")
        return DetectorOutcome(False)

    def _has_multi_redirect_chain(self, context: URLContext) -> DetectorOutcome:
        """Detect chained redirects (>=2 redirect params or nested shortener URLs).
        Attackers chain multiple redirects to evade URL scanners."""
        redirect_count = 0
        nested_urls = []
        for key, values in context.query_params.items():
            key_lower = key.lower()
            if key_lower in self.config.REDIRECT_PARAMS or "redirect" in key_lower or "url" in key_lower:
                for value in values:
                    if self._value_looks_like_url(value):
                        redirect_count += 1
                        nested_urls.append(f"{key}=<url>")

        # Check for nested shortener chains in redirect target
        for key, values in context.query_params.items():
            for value in values:
                try:
                    decoded = unquote(value)
                except Exception:
                    decoded = value
                for shortener in self.config.URL_SHORTENERS:
                    if shortener in decoded:
                        redirect_count += 1
                        nested_urls.append(f"nested_shortener:{shortener}")
                        break

        if redirect_count >= 2:
            return DetectorOutcome(True, context=f"chain={redirect_count};{';'.join(nested_urls[:3])}")
        return DetectorOutcome(False)

    def _is_url_protection_abuse(self, context: URLContext) -> DetectorOutcome:
        """Detect abuse of legitimate URL protection/rewriting services.
        Attackers wrap malicious URLs in Proofpoint, Mimecast, Safe Links, etc."""
        host = context.host
        if not host:
            return DetectorOutcome(False)

        for protection_domain in self.config.URL_PROTECTION_SERVICES:
            if host == protection_domain or host.endswith(f".{protection_domain}"):
                return DetectorOutcome(True, context=protection_domain)
        return DetectorOutcome(False)

    def _is_homoglyph_domain(self, context: URLContext) -> DetectorOutcome:
        """Detect visually confusable (homoglyph) domain names beyond punycode.
        Uses Cyrillic/Greek → Latin confusable mapping to generate a skeleton
        and checks if it matches any known brand domain."""
        host = context.host
        if not host:
            return DetectorOutcome(False)

        # Only check if the host contains non-ASCII characters
        if all(ord(ch) < 128 for ch in host):
            return DetectorOutcome(False)

        # Build ASCII skeleton by replacing confusable chars
        skeleton = []
        has_confusable = False
        for ch in host:
            mapped = self.config.CONFUSABLE_CHAR_MAP.get(ch)
            if mapped:
                skeleton.append(mapped)
                has_confusable = True
            else:
                skeleton.append(ch)

        if not has_confusable:
            return DetectorOutcome(False)

        skeleton_str = "".join(skeleton).lower()

        # Check if skeleton matches any brand token
        for brand in self.config.POPULAR_BRANDS:
            if brand in skeleton_str:
                return DetectorOutcome(True, context=f"homoglyph->{brand};skeleton={skeleton_str}")
        return DetectorOutcome(False)

    def _is_dns_wildcard_subdomain(self, context: URLContext) -> DetectorOutcome:
        """Detect wildcard DNS abuse with infinite unique subdomains.
        Attackers use wildcard DNS to generate unique URLs per victim,
        defeating blocklist-based detection."""
        subdomain = context.subdomain
        if not subdomain:
            return DetectorOutcome(False)

        labels = subdomain.split(".")
        label_count = len(labels)

        # >=3 subdomain labels with random-looking components
        if label_count >= 3:
            random_labels = sum(1 for label in labels if self._looks_random(label) or len(label) > 20)
            if random_labels >= 1:
                return DetectorOutcome(True, context=f"depth={label_count};random_labels={random_labels}")

        # Very long subdomain string (likely auto-generated)
        if len(subdomain) > 60 and label_count >= 2:
            return DetectorOutcome(True, context=f"long_subdomain={len(subdomain)}")

        return DetectorOutcome(False)

    def _is_captcha_shield(self, context: URLContext) -> DetectorOutcome:
        """Detect CAPTCHA services used to shield phishing pages from scanners.
        Phishing sites deploy CAPTCHAs to block automated security analysis."""
        if context.registrable_domain in self.config.BRAND_DOMAINS:
            return DetectorOutcome(False)

        haystack = f"{context.path.lower()}?{context.query.lower()}"

        captcha_hit = None
        for keyword in self.config.CAPTCHA_KEYWORD_LIST:
            if keyword in haystack:
                captcha_hit = keyword
                break
        if not captcha_hit:
            return DetectorOutcome(False)

        # CAPTCHA + credential/suspicious keyword co-occurrence = phishing shield
        for cred_kw in self.config.CREDENTIAL_FORM_KEYWORD_LIST:
            if cred_kw in haystack:
                return DetectorOutcome(True, context=f"{captcha_hit}+{cred_kw}")

        for sus_kw in ("verify", "update", "confirm", "secure", "account"):
            if sus_kw in haystack:
                return DetectorOutcome(True, context=f"{captcha_hit}+{sus_kw}")

        return DetectorOutcome(False)

    def _is_compromised_cms(self, context: URLContext) -> DetectorOutcome:
        """Detect URLs pointing to CMS exploit paths frequently hosting phishing kits.
        Expands beyond basic wp-login to cover Joomla, Drupal, Magento, OWA, etc."""
        if context.registrable_domain in self.config.BRAND_DOMAINS:
            return DetectorOutcome(False)

        path_lower = context.path.lower()
        if not path_lower or path_lower == "/":
            return DetectorOutcome(False)

        for cms_path in self.config.CMS_EXPLOIT_PATH_LIST:
            if cms_path.lower() in path_lower:
                # Extra confidence: CMS path + suspicious file extension
                filename = path_lower.rsplit("/", 1)[-1]
                if any(filename.endswith(ext) for ext in (".html", ".htm", ".php", ".asp", ".aspx", ".jsp")):
                    return DetectorOutcome(True, context=f"{cms_path}+{filename}")
                return DetectorOutcome(True, context=cms_path)

        return DetectorOutcome(False)

    def _is_qr_code_phishing(self, context: URLContext) -> DetectorOutcome:
        """Detect QR code generation APIs used in quishing attacks.
        QR codes bypass email scanners since the URL is image-embedded."""
        host = context.host
        if not host:
            return DetectorOutcome(False)

        for qr_domain in self.config.QR_API_DOMAIN_LIST:
            if host == qr_domain or host.endswith(f".{qr_domain}"):
                return DetectorOutcome(True, context=qr_domain)

        # Check for QR generation query patterns
        query_lower = context.query.lower()
        if "cht=qr" in query_lower or "qr_code" in query_lower or "qrcode" in query_lower:
            return DetectorOutcome(True, context="qr_query_param")

        return DetectorOutcome(False)

    def _is_crypto_scam(self, context: URLContext) -> DetectorOutcome:
        """Detect cryptocurrency scam URLs: wallet-connect, airdrop claims,
        seed-phrase harvesting, fake DEX/CEX pages."""
        if context.registrable_domain in self.config.BRAND_DOMAINS:
            return DetectorOutcome(False)

        haystack = f"{context.path.lower()}/{context.query.lower()}"

        for keyword in self.config.CRYPTO_SCAM_KEYWORD_LIST:
            if keyword in haystack:
                return DetectorOutcome(True, context=keyword)
        return DetectorOutcome(False)

    def _is_dynamic_dns(self, context: URLContext) -> DetectorOutcome:
        """Detect domains hosted on dynamic DNS providers.
        DDNS lets attackers rapidly spin up phishing infrastructure at zero cost."""
        registrable = context.registrable_domain
        if not registrable:
            return DetectorOutcome(False)

        for ddns in self.config.DDNS_PROVIDER_DOMAINS:
            if registrable == ddns or registrable.endswith(f".{ddns}"):
                return DetectorOutcome(True, context=ddns)
            # Also check if host ends with ddns (subdomain of ddns)
            if context.host and context.host.endswith(f".{ddns}"):
                return DetectorOutcome(True, context=ddns)
        return DetectorOutcome(False)

    def _has_embedded_email_target(self, context: URLContext) -> DetectorOutcome:
        """Detect pre-filled email addresses in URL params.
        Personalized phishing pre-fills victim's email in fake login forms."""
        for key, values in context.query_params.items():
            key_lower = key.lower()
            if key_lower in self.config.EMAIL_PARAM_NAME_LIST:
                for value in values:
                    if self._email_pattern.search(value):
                        # Mask the email for privacy in context
                        parts = value.split("@")
                        if len(parts) == 2:
                            masked = f"{parts[0][:2]}***@{parts[1]}"
                        else:
                            masked = "<email>"
                        return DetectorOutcome(True, context=f"{key}={masked}")
        return DetectorOutcome(False)

    def _has_subdomain_depth_abuse(self, context: URLContext) -> DetectorOutcome:
        """Detect excessive subdomain depth (>3 labels).
        Long subdomain chains push the real domain off-screen in address bars."""
        subdomain = context.subdomain
        if not subdomain:
            return DetectorOutcome(False)

        labels = subdomain.split(".")
        depth = len(labels)

        if depth > 3:
            return DetectorOutcome(True, context=f"depth={depth};sub={subdomain[:50]}")
        return DetectorOutcome(False)

    def _is_publishing_platform_abuse(self, context: URLContext) -> DetectorOutcome:
        """Detect phishing hosted on trusted publishing/content creation platforms.
        These platforms have high trust scores and often bypass corporate filters."""
        host = context.host
        if not host:
            return DetectorOutcome(False)

        matched_platform = None
        for platform in self.config.PUBLISHING_PLATFORM_LIST:
            if host == platform or host.endswith(f".{platform}"):
                matched_platform = platform
                break

        if not matched_platform:
            return DetectorOutcome(False)

        # Only flag if combined with suspicious/credential keywords
        haystack = f"{context.path.lower()}?{context.query.lower()}"
        for keyword in self.config.CREDENTIAL_FORM_KEYWORD_LIST:
            if keyword in haystack:
                return DetectorOutcome(True, context=f"{matched_platform}+{keyword}")
        for keyword in ("verify", "update", "confirm", "secure", "account", "bank", "payment", "wallet"):
            if keyword in haystack:
                return DetectorOutcome(True, context=f"{matched_platform}+{keyword}")

        return DetectorOutcome(False)

    def _is_lookalike_tld_swap(self, context: URLContext) -> DetectorOutcome:
        """Detect brand domain + wrong TLD (e.g., paypal.xyz, amazon.shop).
        Domain text is visually correct but TLD is swapped to an abusive one."""
        domain = context.domain.lower()
        suffix = context.suffix.lower()
        if not domain or not suffix:
            return DetectorOutcome(False)

        # Check if the domain part exactly matches a known brand
        legitimate_tlds = self.config.BRAND_TLD_MAP.get(domain)
        if legitimate_tlds is None:
            return DetectorOutcome(False)

        # If current TLD is NOT in the brand's legitimate TLD set → flag
        if suffix not in legitimate_tlds:
            return DetectorOutcome(True, context=f"{domain}.{suffix}(expected:{','.join(sorted(legitimate_tlds))})")
        return DetectorOutcome(False)

    def _is_disposable_email_abuse(self, context: URLContext) -> DetectorOutcome:
        """Detect disposable/temporary email service domains in URL parameters.
        Presence indicates automated phishing infrastructure."""
        for key, values in context.query_params.items():
            for value in values:
                val_lower = value.lower()
                for disp_domain in self.config.DISPOSABLE_EMAIL_LIST:
                    if disp_domain in val_lower:
                        return DetectorOutcome(True, context=f"{key}:{disp_domain}")
        # Also check if the host itself is a disposable email service
        if context.registrable_domain in self.config.DISPOSABLE_EMAIL_LIST:
            return DetectorOutcome(True, context=f"host:{context.registrable_domain}")
        return DetectorOutcome(False)

    def _has_urgency_manipulation(self, context: URLContext) -> DetectorOutcome:
        """Detect extreme urgency/time-pressure manipulation in URL path/query.
        Social engineering tactic to force hasty decisions."""
        haystack = f"{context.path.lower()}?{context.query.lower()}"

        for keyword in self.config.URGENCY_KEYWORD_LIST:
            if keyword in haystack:
                return DetectorOutcome(True, context=keyword)
        return DetectorOutcome(False)

    def _is_ipfs_hosting(self, context: URLContext) -> DetectorOutcome:
        """Detect IPFS/decentralized hosting URLs.
        IPFS content is immutable and cannot be easily taken down —
        increasingly used for persistent phishing pages."""
        # Check scheme
        if context.scheme in {"ipfs", "ipns"}:
            return DetectorOutcome(True, context=context.scheme)

        # Check known gateway domains
        host = context.host
        if not host:
            return DetectorOutcome(False)

        for gateway in self.config.IPFS_GATEWAY_LIST:
            if host == gateway or host.endswith(f".{gateway}"):
                return DetectorOutcome(True, context=gateway)

        # Check for IPFS CID patterns in path (Qm... or bafy...)
        if re.search(r"/(?:ipfs|ipns)/(?:Qm[a-zA-Z0-9]{44}|bafy[a-zA-Z0-9]{50,})", context.path):
            return DetectorOutcome(True, context="ipfs_cid_in_path")

        return DetectorOutcome(False)

    def _is_telegram_bot(self, context: URLContext) -> DetectorOutcome:
        """Detect Telegram bot interaction URLs used in scam automation.
        Telegram bots are heavily used in crypto scams and credential exfiltration."""
        host = context.host
        if not host:
            return DetectorOutcome(False)

        # Telegram bot API calls
        if host == "api.telegram.org" and "/bot" in context.path:
            return DetectorOutcome(True, context="telegram_bot_api")

        # Telegram short links to bots
        if host in {"t.me", "telegram.me"}:
            path_lower = context.path.lower()
            if "bot" in path_lower:
                return DetectorOutcome(True, context="telegram_bot_link")
            # Also check for app-specific paths
            if any(token in path_lower for token in ("start=", "startgroup=", "startapp=")):
                return DetectorOutcome(True, context="telegram_bot_start")
            # Check query for start commands
            if context.query and any(token in context.query.lower() for token in ("start=", "startgroup=", "startapp=")):
                return DetectorOutcome(True, context="telegram_bot_start_param")

        return DetectorOutcome(False)

    def _is_nested_encoding_bypass(self, context: URLContext) -> DetectorOutcome:
        """Detect URLs with extreme encoding depth (Refinement v6.3).
        URLs with 5+ layers of percent-encoding are used to bypass scanners and bypass decoders."""
        # Multi-pass decode to check depth
        current = context.raw_url
        passes = 0
        for _ in range(12):
            try:
                decoded = unquote(current)
                if decoded == current:
                    break
                current = decoded
                passes += 1
            except Exception:
                break
        
        if passes >= 5:
            return DetectorOutcome(True, context=f"passes={passes}")
        return DetectorOutcome(False)

    def _is_deep_domain_stacking(self, context: URLContext) -> DetectorOutcome:
        """Detect extreme domain stacking (Refinement v6.3).
        More than 5 labels in the host part indicates complex domain stacking evasion."""
        host = context.host
        if not host:
            return DetectorOutcome(False)
        
        dots = host.count('.')
        if dots >= 5:
            return DetectorOutcome(True, context=f"labels={dots + 1}")
        return DetectorOutcome(False)

    def _is_phishing_bypass_path(self, context: URLContext) -> DetectorOutcome:
        """Detect known bypass paths like Cloudflare phish-bypass or atok (Refinement v6.3)."""
        path_lower = context.path.lower()
        if "cdn-cgi/phish-bypass" in path_lower:
            return DetectorOutcome(True, context="cloudflare_phish_bypass")
        
        # Check for atok or similar high-entropy bypass tokens in query
        for key in context.query_params:
            if key.lower() in {"atok", "bypass_token", "phish_bypass"}:
                return DetectorOutcome(True, context=f"param:{key}")
        
        return DetectorOutcome(False)

    def _is_public_document_abuse(self, context: URLContext) -> DetectorOutcome:
        """Detect Google Docs embedding abuse patterns (Refinement v6.3).
        Example: docs.google.com/document/d/e/2pacx-1v..."""
        host = context.host
        if not host:
            return DetectorOutcome(False)
            
        if "docs.google.com" in host or "forms.google.com" in host:
            path_lower = context.path.lower()
            if "/e/2pacx-" in path_lower:
                return DetectorOutcome(True, context="public_embed_form")
            if "/pub" in path_lower:
                return DetectorOutcome(True, context="public_doc_publish")
        
        return DetectorOutcome(False)

    def _is_protocol_keyword_subdomain(self, context: URLContext) -> DetectorOutcome:
        """Detect protocol keywords used as subdomains (Refinement v6.3).
        Example: https.paypal.com.secure-update.net"""
        subdomain = context.subdomain.lower()
        if not subdomain:
            return DetectorOutcome(False)
            
        labels = set(subdomain.split('.'))
        dangerous_keywords = {"https", "http", "www", "secure", "ssl"}
        
        found = dangerous_keywords.intersection(labels)
        if found:
            return DetectorOutcome(True, context=f"keyword:{list(found)[0]}")
            
        return DetectorOutcome(False)

    def _has_non_english_chars(self, context: URLContext) -> DetectorOutcome:
        """Flags URLs explicitly containing visible non-English (non-ASCII) characters or punycode."""
        if not context.raw_url:
            return DetectorOutcome(False)
            
        # Check raw url for raw non-ascii (like Cyrillic, Chinese, Arabic)
        if self._unicode_pattern.search(context.raw_url):
            return DetectorOutcome(True, context="raw_non_ascii_chars")
        
        # Check unquoted URL for percent-encoded non-english chars
        try:
            unquoted = unquote(context.raw_url, encoding='utf-8', errors='strict')
            if unquoted != context.raw_url and self._unicode_pattern.search(unquoted):
                return DetectorOutcome(True, context="percent_encoded_non_ascii")
        except UnicodeDecodeError:
            pass # Invalid UTF-8 naturally gets handled by malform detectors

        # Punycode domains inherently represent non-English structures
        if context.parsed.hostname and self._punycode_pattern.search(context.parsed.hostname):
            return DetectorOutcome(True, context="punycode_idna_domain")

        return DetectorOutcome(False)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _value_looks_like_url(self, value: str) -> bool:
        candidate = (value or "").strip().lower()
        if candidate.startswith(("http://", "https://", "ftp://", "//")):
            return True
        try:
            decoded = unquote(candidate)
        except Exception:
            decoded = candidate
        return decoded.startswith(("http://", "https://", "//"))

    def _looks_random(self, value: str) -> bool:
        if not value:
            return False
        entropy = self._shannon_entropy(value)
        return entropy > 3.5 and sum(ch.isdigit() for ch in value) + sum(ch.isalpha() for ch in value) > 10

    def _brand_hits(self, haystack: str) -> Set[str]:
        hits: Set[str] = set()
        text = haystack.lower()
        for brand in self.config.POPULAR_BRANDS:
            pattern = self._brand_regex_cache.get(brand)
            if pattern is None:
                if len(brand) <= 4:
                    regex = rf"(?<![a-z0-9]){re.escape(brand)}(?![a-z0-9])"
                else:
                    regex = rf"(?<![a-z0-9]){re.escape(brand)}(?![a-z0-9])|{re.escape(brand)}(?=[\./\-]|$)"
                pattern = re.compile(regex)
                self._brand_regex_cache[brand] = pattern
            if pattern.search(text):
                hits.add(brand)
        return hits

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        previous_row = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            current_row = [i]
            for j, cb in enumerate(b, start=1):
                insertions = previous_row[j] + 1
                deletions = current_row[j - 1] + 1
                substitutions = previous_row[j - 1] + (ca != cb)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]




@dataclass
class URLContext:
    """Pre-parsed URL components shared across detectors."""

    raw_url: str
    normalized_url: str
    scheme: str
    host: str
    port: Optional[int]
    path: str
    query: str
    fragment: str
    subdomain: str
    domain: str
    suffix: str
    registrable_domain: str
    full_domain: str
    query_params: Dict[str, List[str]]
    path_segments: List[str]
    url_length: int
    path_depth: int
    num_params: int
    entropy: float
    digit_ratio: float
    structural_malformed: bool = False
    parse_error: Optional[str] = None


@dataclass
class DetectorOutcome:
    """Single detector outcome with optional context and scoring override."""

    triggered: bool
    context: Optional[str] = None
    score_component: Optional[float] = None

@dataclass
class AnalysisDetails:
    """Structured analysis output used for scoring and explainability."""

    flags: Dict[str, bool]
    contexts: Dict[str, Optional[str]]
    score_components: Dict[str, float]
    detection_score: float
    detection_explain: str
    active_categories: List[str]



# ---------------------------------------------------------------------------
# LEVEL 3 – METADATA & CONSTANTS
# ---------------------------------------------------------------------------
class URLCategory:
    """Metadata for each spreadsheet-backed URL category."""

    CATEGORIES: Dict[str, Dict[str, str]] = {
        "Non_English_Characters_URL": {
            "description": "Contains non-English (non-ASCII) characters or punycode indicating a foreign language targeted lure.",
            "severity": "MEDIUM",
            "folder": "Non_English_Characters_URL",
            "friendly": "NonEnglishChars",
        },
        "TypoSquatting_URL": {
            "description": "Misspells trusted domains to capture credentials.",
            "severity": "CRITICAL",
            "folder": "TypoSquatting_URL",
            "friendly": "Typosquatting",
        },
        "Shortened_URL": {
            "description": "Uses URL shorteners to hide the destination.",
            "severity": "HIGH",
            "folder": "Shortened_URL",
            "friendly": "Shortener",
        },
        "Punycode_URL": {
            "description": "Uses xn-- punycode for homoglyph spoofing.",
            "severity": "HIGH",
            "folder": "Punycode_URL",
            "friendly": "Punycode",
        },
        "Unicode_URL": {
            "description": "Contains non-ASCII characters in host or path.",
            "severity": "MEDIUM",
            "folder": "Unicode_URL",
            "friendly": "Unicode",
        },
        "Hex_Encoded_URL": {
            "description": "Excessive percent-encoding to hide payloads.",
            "severity": "MEDIUM",
            "folder": "Hex_Encoded_URL",
            "friendly": "HexEncoding",
        },
        "IP_Address_Unusual_Port_URL": {
            "description": "Direct IP usage or suspicious non-standard ports.",
            "severity": "HIGH",
            "folder": "IP_Address_Unusual_Port_URL",
            "friendly": "IPorPort",
        },
        "Decimal_Hex_IP_URL": {
            "description": "IPv4 represented in decimal or hexadecimal form.",
            "severity": "HIGH",
            "folder": "Decimal_Hex_IP_URL",
            "friendly": "ObfuscatedIP",
        },
        "Data_URL": {
            "description": "data: URL embedding inline payload.",
            "severity": "CRITICAL",
            "folder": "Data_URL",
            "friendly": "DataScheme",
        },
        "JavaScript_URL": {
            "description": "javascript: URL executing inline script.",
            "severity": "CRITICAL",
            "folder": "JavaScript_URL",
            "friendly": "JavaScriptScheme",
        },
        "File_URL": {
            "description": "Local file access via file:// or drive path.",
            "severity": "HIGH",
            "folder": "File_URL",
            "friendly": "FileScheme",
        },
        "FTP_SFTP_URL": {
            "description": "FTP/SFTP schemes for data exfiltration.",
            "severity": "MEDIUM",
            "folder": "FTP_SFTP_URL",
            "friendly": "FTP",
        },
        "Blob_URL": {
            "description": "blob: URLs hosting script payloads.",
            "severity": "HIGH",
            "folder": "Blob_URL",
            "friendly": "BlobScheme",
        },
        "Anchor_Fragment_Based_URL": {
            "description": "Fragments hiding secondary HTTP target.",
            "severity": "MEDIUM",
            "folder": "Anchor_Fragment_Based_URL",
            "friendly": "FragmentRedirect",
        },
        "Redirect_URL_Open_Redirect": {
            "description": "Open redirect parameters leaking external URLs.",
            "severity": "HIGH",
            "folder": "Redirect_URL_Open_Redirect",
            "friendly": "OpenRedirect",
        },
        "Suspiciously_Long_Complex_URL": {
            "description": "Overly long URLs with deep paths or parameters.",
            "severity": "MEDIUM",
            "folder": "Suspiciously_Long_Complex_URL",
            "friendly": "LongComplex",
        },
        "Fake_Subdomain_URL": {
            "description": "Brand token in subdomain with mismatched registrable domain.",
            "severity": "HIGH",
            "folder": "Fake_Subdomain_URL",
            "friendly": "FakeSubdomain",
        },
        "Chrome_Internal_URL": {
            "description": "Attempts to mimic browser internal UIs.",
            "severity": "HIGH",
            "folder": "Chrome_Internal_URL",
            "friendly": "ChromeInternal",
        },
        "Suspicious_TLD_URL": {
            "description": "Abuse-prone or contextual TLD usage.",
            "severity": "HIGH",
            "folder": "Suspicious_TLD_URL",
            "friendly": "SuspiciousTLD",
        },
        "Tunneling_URLs_Proxy_Abuse": {
            "description": "Proxy, translate or cache services masking destination.",
            "severity": "HIGH",
            "folder": "Tunneling_URLs_Proxy_Abuse",
            "friendly": "ProxyAbuse",
        },
        "Mobile_AMP_URLs": {
            "description": "Mobile / AMP cloaking layers.",
            "severity": "LOW",
            "folder": "Mobile_AMP_URLs",
            "friendly": "MobileAMP",
        },
        "HasExcessiveParams": {
            "description": "URL contains excessive query parameters.",
            "severity": "MEDIUM",
            "folder": "HasExcessiveParams",
            "friendly": "ExcessiveParams",
        },
        "HasRepeatedSubdomain": {
            "description": "Repeated subdomain components suggesting trickery.",
            "severity": "MEDIUM",
            "folder": "HasRepeatedSubdomain",
            "friendly": "RepeatedSubdomain",
        },
        "IsBrandImpersonation": {
            "description": "Brand token present but registrable domain unauthorised.",
            "severity": "CRITICAL",
            "folder": "IsBrandImpersonation",
            "friendly": "BrandImpersonation",
        },
        "IsDynamicQuery": {
            "description": "Dynamic query parameters requesting sensitive data.",
            "severity": "MEDIUM",
            "folder": "IsDynamicQuery",
            "friendly": "DynamicQuery",
        },
        "IsGeoLocationSpecific": {
            "description": "Geo-targeted lure using specific TLDs and keywords.",
            "severity": "LOW",
            "folder": "IsGeoLocationSpecific",
            "friendly": "GeoSpecific",
        },
        "IsLanguageSpecific": {
            "description": "Language-aware lure using locale tokens.",
            "severity": "LOW",
            "folder": "IsLanguageSpecific",
            "friendly": "LanguageSpecific",
        },
        "IsObfuscatedURL": {
            "description": "High entropy or obfuscated path structures.",
            "severity": "HIGH",
            "folder": "IsObfuscatedURL",
            "friendly": "Obfuscated",
        },
        "IsSessionBased": {
            "description": "Session tokens exposed in the URL.",
            "severity": "MEDIUM",
            "folder": "IsSessionBased",
            "friendly": "SessionToken",
        },
        "IsSuspiciousFileType": {
            "description": "Executable or risky file download extensions.",
            "severity": "CRITICAL",
            "folder": "IsSuspiciousFileType",
            "friendly": "SuspiciousFile",
        },
        "IsSuspiciousKeyword": {
            "description": "High-risk social engineering keywords.",
            "severity": "MEDIUM",
            "folder": "IsSuspiciousKeyword",
            "friendly": "SuspiciousKeyword",
        },
        "IsMaliciousPattern": {
            "description": "Direct evidence of SQLi/XSS/command patterns.",
            "severity": "CRITICAL",
            "folder": "IsMaliciousPattern",
            "friendly": "MaliciousPattern",
        },
        "IsWebAppPath": {
            "description": "Access attempt to webapp login/admin paths.",
            "severity": "HIGH",
            "folder": "IsWebAppPath",
            "friendly": "WebAppPath",
        },
        "Structural_Malformation_URL": {
            "description": "Malformed or structurally invalid URLs (parser failures, broken schemes).",
            "severity": "HIGH",
            "folder": "Structural_Malformation_URL",
            "friendly": "StructuralMalformed",
        },
        "Very_Short_URL": {
            "description": "Unusually short URLs often used for masking or simple lures.",
            "severity": "LOW",
            "folder": "Very_Short_URL",
            "friendly": "VeryShort",
        },
        "Non_Alpha_Start_URL": {
            "description": "Hostname starts with non-alphabetic character (digit, IP, hex, etc.).",
            "severity": "MEDIUM",
            "folder": "Non_Alpha_Start_URL",
            "friendly": "NonAlphaStartHost",
        },
        "Cloud_Hosting_Abuse_URL": {
            "description": "URL hosted on cloud storage/file hosting platforms often abused for phishing.",
            "severity": "HIGH",
            "folder": "Cloud_Hosting_Abuse_URL",
            "friendly": "CloudHostingAbuse",
        },
        # --- NEW PHISHING DETECTOR METADATA ---
        "Credential_Harvesting_Form_URL": {
            "description": "URLs mimicking credential forms on non-brand domains.",
            "severity": "CRITICAL",
            "folder": "Credential_Harvesting_Form_URL",
            "friendly": "CredentialForm",
        },
        "Multi_Redirect_Chain_URL": {
            "description": "Chained redirects or nested shortener URLs to evade scanners.",
            "severity": "HIGH",
            "folder": "Multi_Redirect_Chain_URL",
            "friendly": "MultiRedirectChain",
        },
        "URL_Protection_Service_Abuse": {
            "description": "Abuse of legitimate URL protection/rewriting services.",
            "severity": "HIGH",
            "folder": "URL_Protection_Service_Abuse",
            "friendly": "URLProtectionAbuse",
        },
        "Homoglyph_Domain_URL": {
            "description": "Visually confusable (homoglyph) domain names beyond punycode.",
            "severity": "CRITICAL",
            "folder": "Homoglyph_Domain_URL",
            "friendly": "HomoglyphDomain",
        },
        "DNS_Wildcard_Infinite_Subdomain": {
            "description": "Wildcard DNS abuse with infinite unique subdomains.",
            "severity": "HIGH",
            "folder": "DNS_Wildcard_Infinite_Subdomain",
            "friendly": "InfiniteSubdomain",
        },
        "CAPTCHA_Shield_URL": {
            "description": "CAPTCHA services used to shield phishing pages from scanners.",
            "severity": "HIGH",
            "folder": "CAPTCHA_Shield_URL",
            "friendly": "CaptchaShield",
        },
        "Compromised_CMS_URL": {
            "description": "URLs pointing to CMS exploit paths frequently hosting phishing kits.",
            "severity": "HIGH",
            "folder": "Compromised_CMS_URL",
            "friendly": "CompromisedCMS",
        },
        "QR_Code_Phishing_URL": {
            "description": "QR code generation APIs used in quishing attacks.",
            "severity": "MEDIUM",
            "folder": "QR_Code_Phishing_URL",
            "friendly": "QuishingAPI",
        },
        "Cryptocurrency_Scam_URL": {
            "description": "Cryptocurrency scam URLs: wallet-connect, seed-phrase harvesting.",
            "severity": "MEDIUM",
            "folder": "Cryptocurrency_Scam_URL",
            "friendly": "CryptoScam",
        },
        "Dynamic_DNS_URL": {
            "description": "Domains hosted on dynamic DNS providers.",
            "severity": "MEDIUM",
            "folder": "Dynamic_DNS_URL",
            "friendly": "DynamicDNS",
        },
        "Embedded_Email_Target_URL": {
            "description": "Pre-filled email addresses in URL parameters.",
            "severity": "MEDIUM",
            "folder": "Embedded_Email_Target_URL",
            "friendly": "EmbeddedEmail",
        },
        "Subdomain_Depth_Abuse_URL": {
            "description": "Excessive subdomain depth to push real domain out of view.",
            "severity": "MEDIUM",
            "folder": "Subdomain_Depth_Abuse_URL",
            "friendly": "SubdomainDepthAbuse",
        },
        "Digital_Publishing_Platform_Abuse": {
            "description": "Phishing hosted on trusted publishing/content creation platforms.",
            "severity": "MEDIUM",
            "folder": "Digital_Publishing_Platform_Abuse",
            "friendly": "PublishingPlatformAbuse",
        },
        "Lookalike_TLD_Swap_URL": {
            "description": "Brand domain text matches, but TLD is swapped to an abusive one.",
            "severity": "MEDIUM",
            "folder": "Lookalike_TLD_Swap_URL",
            "friendly": "LookalikeTLDSwap",
        },
        "Disposable_Email_Abuse_URL": {
            "description": "Disposable/temporary email service domains in parameters.",
            "severity": "LOW",
            "folder": "Disposable_Email_Abuse_URL",
            "friendly": "DisposableEmailAbuse",
        },
        "Social_Engineering_Urgency_URL": {
            "description": "Extreme urgency/time-pressure manipulation in URL.",
            "severity": "LOW",
            "folder": "Social_Engineering_Urgency_URL",
            "friendly": "UrgencyManipulation",
        },
        "IPFS_Decentralized_Hosting_URL": {
            "description": "IPFS/decentralized hosting URLs for persistent phishing.",
            "severity": "LOW",
            "folder": "IPFS_Decentralized_Hosting_URL",
            "friendly": "IPFSHosting",
        },
        "Telegram_Bot_URL": {
            "description": "Telegram bot interaction URLs used in scam automation.",
            "severity": "LOW",
            "folder": "Telegram_Bot_URL",
            "friendly": "TelegramBot",
        },
        "Nested_Encoding_Bypass_URL": {
            "description": "URLs with extreme percent-encoding depth (5+ layers).",
            "severity": "HIGH",
            "folder": "Nested_Encoding_Bypass_URL",
            "friendly": "NestedEncoding",
        },
        "Deep_Domain_Stacking_URL": {
            "description": "Extreme domain stacking with 5+ labels in the host.",
            "severity": "HIGH",
            "folder": "Deep_Domain_Stacking_URL",
            "friendly": "DomainStacking",
        },
        "Phishing_Bypass_Path_URL": {
            "description": "Known phishing scanner bypass paths and tokens.",
            "severity": "CRITICAL",
            "folder": "Phishing_Bypass_Path_URL",
            "friendly": "PhishBypass",
        },
        "Public_Document_Abuse_URL": {
            "description": "Abuse of public document embedding and publishing features.",
            "severity": "HIGH",
            "folder": "Public_Document_Abuse_URL",
            "friendly": "PubDocAbuse",
        },
        "Protocol_Keyword_Subdomain_URL": {
            "description": "Protocol keywords (https, www) used as subdomains for deception.",
            "severity": "HIGH",
            "folder": "Protocol_Keyword_Subdomain_URL",
            "friendly": "ProtoSubdomain",
        },
    }

    @classmethod
    def categories(cls) -> List[str]:
        return list(cls.CATEGORIES.keys())

    @classmethod
    def friendly_name(cls, category: str) -> str:
        return cls.CATEGORIES.get(category, {}).get("friendly", category)

    @classmethod
    def folder_name(cls, category: str) -> str:
        info = cls.CATEGORIES.get(category, {})
        return info.get("folder", category)
    
# ---------------------------------------------------------------------------
# Visualization stub (legacy CLI compatibility)
# ---------------------------------------------------------------------------
class URLCategoryVisualizer:
    """Placeholder visualizer preserving CLI compatibility."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def generate_all_visualizations(self) -> None:
        logging.info("Visualization step not implemented in this refactor; skipping.")

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="URL categorization toolkit aligned with the URLs Categories spreadsheet.")

    parser.add_argument("--input", "-i", type=str, default="data.csv", help="Input CSV containing URLs.")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to store categorised outputs.")
    parser.add_argument("--url-column", type=str, default="input", help="Column name containing URLs.")
    parser.add_argument("--label-column", type=str, default="label", help="Column name containing labels.")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Chunk size for streaming processing.")
    parser.add_argument("--workers", type=int, default=1, help="Thread workers for URL analysis.")
    parser.add_argument("--no-unmatched", action="store_true", help="Do not export unmatched URLs.")
    parser.add_argument("--summary-report", type=str, default="categorization_report.txt", help="Summary report name.")
    parser.add_argument("--visualize", action="store_true", help="Generate visualisations (stub).")
    parser.add_argument("--visualize-only", action="store_true", help="Only run the visualization step.")
    args = parser.parse_args()

    config = CategoryConfig(
        INPUT_FILE=args.input,
        OUTPUT_DIR=resolve_project_path(args.output),
        SUMMARY_REPORT=args.summary_report,
        URL_COLUMN=args.url_column,
        LABEL_COLUMN=args.label_column,
        SAVE_UNMATCHED=not args.no_unmatched,
        CHUNK_SIZE=args.chunk_size,
        WORKERS=args.workers,
    )

    if not args.visualize_only:
        categorizer = URLCategorizer(config)
        categorizer.categorize_dataset()

    if args.visualize or args.visualize_only:
        visualizer = URLCategoryVisualizer(config.OUTPUT_DIR)
        visualizer.generate_all_visualizations()


# 0. argparse → builds CategoryConfig → fires URLCategorizer
if __name__ == "__main__":
    main()





# python urls_cate_V7.py --input "C:\Users\HP\Desktop\DataPrep8\2_MiniLM_Hybrid_FF_V4\DATA\3_LNU_Phish1.csv"
