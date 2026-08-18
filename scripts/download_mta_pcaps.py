"""Download Malware-Traffic-Analysis.net PCAP archives.

The site publishes password-protected zip archives.  Newer posts use
``infected_YYYYMMDD`` and older posts commonly use ``infected``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


BASE_URL = "https://www.malware-traffic-analysis.net/"
DEFAULT_YEARS = range(2013, 2027)
PCAP_EXTS = (".pcap", ".pcapng", ".cap")
DATE_RE = re.compile(r"/(?P<year>20\d\d)/(?P<month>\d\d)/(?P<day>\d\d)/")
NAME_DATE_RE = re.compile(r"(?P<year>20\d\d)-(?P<month>\d\d)-(?P<day>\d\d)")


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.links.append(value)


@dataclass(frozen=True)
class DownloadJob:
    article_url: str
    archive_url: str
    post_date: str


def fetch_text(url: str, timeout: int, retries: int = 3) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "rdsynth-mta-pcap-downloader/1.0"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == retries:
                raise
            time.sleep(min(15, 2**attempt))
    raise RuntimeError(f"unreachable retry state for {url}")


def iter_links(url: str, timeout: int) -> list[str]:
    parser = LinkParser()
    parser.feed(fetch_text(url, timeout))
    return [urllib.parse.urljoin(url, link.strip()) for link in parser.links]


def post_date_from_url(url: str) -> str:
    match = DATE_RE.search(url)
    if not match:
        return "unknown"
    return f"{match.group('year')}{match.group('month')}{match.group('day')}"


def discover_jobs(years: list[int], timeout: int, workers: int) -> list[DownloadJob]:
    article_urls: set[str] = set()
    for year in years:
        index_url = urllib.parse.urljoin(BASE_URL, f"{year}/index.html")
        for link in iter_links(index_url, timeout):
            parsed = urllib.parse.urlparse(link)
            if re.search(rf"/{year}/\d\d/\d\d/index\.html$", parsed.path):
                article_urls.add(link)

    jobs: dict[str, DownloadJob] = {}

    def article_pcap_links(article_url: str) -> tuple[str, list[str]]:
        return article_url, iter_links(article_url, timeout)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(article_pcap_links, article_url) for article_url in sorted(article_urls)]
        for future in concurrent.futures.as_completed(futures):
            try:
                article_url, links = future.result()
            except Exception as exc:  # noqa: BLE001 - one unavailable article should not stop discovery.
                print(f"WARNING: article discovery failed: {exc}", file=sys.stderr, flush=True)
                continue
            for link in links:
                name = urllib.parse.unquote(Path(urllib.parse.urlparse(link).path).name).lower()
                if any(ext in name for ext in PCAP_EXTS) and name.endswith((".zip", ".pcap", ".pcapng", ".cap")):
                    jobs[link] = DownloadJob(
                        article_url=article_url,
                        archive_url=link,
                        post_date=post_date_from_url(article_url),
                    )
    return [jobs[url] for url in sorted(jobs)]


def safe_name(name: str) -> str:
    name = urllib.parse.unquote(name)
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name).strip(" .")


def download(url: str, dest: Path, timeout: int, retries: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": "rdsynth-mta-pcap-downloader/1.0"}
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt == retries:
                raise RuntimeError(f"download failed after {retries} attempts: {exc}") from exc
            time.sleep(min(30, 2**attempt))


def extract_pcaps(zip_path: Path, out_dir: Path, post_date: str) -> list[Path]:
    extracted: list[Path] = []
    password_dates = []
    if post_date != "unknown":
        password_dates.append(post_date)
    name_match = NAME_DATE_RE.search(zip_path.name)
    if name_match:
        password_dates.append(f"{name_match.group('year')}{name_match.group('month')}{name_match.group('day')}")
    passwords = [f"infected_{date}".encode("utf-8") for date in dict.fromkeys(password_dates)]
    passwords.append(b"infected")
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            info for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith(PCAP_EXTS)
        ]
        for info in members:
            target = out_dir / safe_name(Path(info.filename).name)
            if target.exists() and target.stat().st_size == info.file_size:
                extracted.append(target)
                continue
            last_error: Exception | None = None
            for password in passwords:
                try:
                    with archive.open(info, pwd=password) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
                    extracted.append(target)
                    last_error = None
                    break
                except RuntimeError as exc:
                    last_error = exc
                    if target.exists():
                        target.unlink()
            if last_error is not None:
                raise RuntimeError(f"could not extract {info.filename}: {last_error}") from last_error
    return extracted


def existing_expected_pcap(archive_name: str, out_dir: Path) -> Path | None:
    lower = archive_name.lower()
    if not lower.endswith(".zip"):
        return None
    expected = archive_name[:-4]
    if not expected.lower().endswith(PCAP_EXTS):
        return None
    path = out_dir / expected
    return path if path.exists() and path.stat().st_size > 0 else None


def write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "status",
                "post_date",
                "article_url",
                "archive_url",
                "archive_name",
                "extracted_files",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def job_from_manifest_row(row: dict[str, str]) -> DownloadJob:
    return DownloadJob(
        article_url=row["article_url"],
        archive_url=row["archive_url"].strip(),
        post_date=row["post_date"],
    )


def process_job(job: DownloadJob, args: argparse.Namespace) -> dict[str, str]:
    archive_name = safe_name(Path(urllib.parse.urlparse(job.archive_url).path).name)
    archive_path = args.archive_dir / archive_name
    status = "success"
    message = ""
    extracted_files: list[Path] = []
    try:
        lower = archive_name.lower()
        if lower.endswith(PCAP_EXTS):
            target = args.out_dir / archive_name
            if not target.exists():
                download(job.archive_url, target, args.timeout, args.retries)
            extracted_files = [target]
        else:
            expected = existing_expected_pcap(archive_name, args.out_dir)
            if expected is not None:
                extracted_files = [expected]
                message = "existing expected PCAP reused"
            else:
                if not archive_path.exists():
                    download(job.archive_url, archive_path, args.timeout, args.retries)
                extracted_files = extract_pcaps(archive_path, args.out_dir, job.post_date)
                if not extracted_files:
                    status = "no-pcap-member"
                    message = "archive had no pcap/pcapng/cap member"
                if not args.keep_archives and archive_path.exists():
                    archive_path.unlink()
    except Exception as exc:  # noqa: BLE001 - manifest records per-file failure and continues.
        status = "failed"
        message = str(exc)
    return {
        "status": status,
        "post_date": job.post_date,
        "article_url": job.article_url,
        "archive_url": job.archive_url,
        "archive_name": archive_name,
        "extracted_files": ";".join(str(path) for path in extracted_files),
        "message": message,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/PCAPs/malicious"))
    parser.add_argument("--manifest", type=Path, default=Path("data/PCAPs/malicious/mta_manifest.csv"))
    parser.add_argument("--years", default="2013-2026", help="Year list/range, e.g. 2019,2020-2026")
    parser.add_argument("--keep-archives", action="store_true", help="Keep downloaded zip files under --archive-dir")
    parser.add_argument("--archive-dir", type=Path, default=Path("outputs/cache/mta_archives"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--download-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def parse_years(spec: str) -> list[int]:
    years: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            years.update(range(start, end + 1))
        else:
            years.add(int(part))
    invalid = [year for year in years if year not in DEFAULT_YEARS]
    if invalid:
        raise ValueError(f"years outside supported range 2013-2026: {invalid}")
    return sorted(years)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    years = parse_years(args.years)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.archive_dir.mkdir(parents=True, exist_ok=True)

    if args.retry_manifest:
        previous_rows = read_manifest(args.retry_manifest)
        jobs = [
            job_from_manifest_row(row)
            for row in previous_rows
            if row.get("status") != "success"
        ]
        print(f"Retrying {len(jobs)} non-success jobs from {args.retry_manifest}", flush=True)
    else:
        print(f"Discovering MTA PCAP links for years: {years[0]}-{years[-1]}", flush=True)
        jobs = discover_jobs(years, args.timeout, args.workers)
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"Discovered {len(jobs)} candidate PCAP archives/direct files", flush=True)

    if args.dry_run:
        rows: list[dict[str, str]] = []
        for idx, job in enumerate(jobs, 1):
            archive_name = safe_name(Path(urllib.parse.urlparse(job.archive_url).path).name)
            print(f"[{idx}/{len(jobs)}] {archive_name}", flush=True)
            rows.append({
                "status": "dry-run",
                "post_date": job.post_date,
                "article_url": job.article_url,
                "archive_url": job.archive_url,
                "archive_name": archive_name,
                "extracted_files": "",
                "message": "",
            })
        write_manifest(args.manifest, rows)
        print(f"Done. success={len(rows)}, no_pcap_member=0, failed=0", flush=True)
        return 0

    rows = []
    if args.download_workers <= 1:
        for idx, job in enumerate(jobs, 1):
            archive_name = safe_name(Path(urllib.parse.urlparse(job.archive_url).path).name)
            print(f"[{idx}/{len(jobs)}] {archive_name}", flush=True)
            rows.append(process_job(job, args))
            write_manifest(args.manifest, rows)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.download_workers) as executor:
            future_to_job = {executor.submit(process_job, job, args): job for job in jobs}
            completed = 0
            for future in concurrent.futures.as_completed(future_to_job):
                completed += 1
                row = future.result()
                print(f"[{completed}/{len(jobs)}] {row['status']} {row['archive_name']}", flush=True)
                rows.append(row)
                write_manifest(args.manifest, rows)

    write_manifest(args.manifest, rows)
    failed = sum(1 for row in rows if row["status"] == "failed")
    no_pcap = sum(1 for row in rows if row["status"] == "no-pcap-member")
    print(f"Done. success={len(rows) - failed - no_pcap}, no_pcap_member={no_pcap}, failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
