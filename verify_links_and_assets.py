#!/usr/bin/env python3
"""
Zevbuild E2E Link & Static Asset Verification Suite
===================================================
A comprehensive, zero-external-dependency crawler and test harness designed to
audit HTML link integrity, DOM fragment anchors, static asset resolution,
and cross-page interface contracts across the entire Zevbuild repository.

Architecture & Test Tiers (PROJECT.md Compliance):
--------------------------------------------------
- Tier 1: Feature Coverage (relative paths, asset loads, entry points)
- Tier 2: Boundary & Corner Cases (empty values, anchors, query strings, ../ traversal)
- Tier 3: Cross-Feature Combinations (hub-to-tool routing, tool-to-hub back links, 404 smart routing)
- Tier 4: Real-World Workload Scenarios (new visitor walkthrough, explorer roundtrip, 404 recovery, outbound link audit)

Standard Library Only:
----------------------
Uses html.parser, urllib.request, urllib.parse, pathlib, json, re, concurrent.futures.
No pip or npm dependencies required.

Exit Codes:
-----------
- 0: All checked tiers passed with zero defects.
- 1: Verification failed (broken internal paths, missing assets, broken anchors, or dead outbound links).
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Configuration Constants
DEFAULT_TIMEOUT = 10
MAX_EXTERNAL_WORKERS = 8
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
    "(ZevbuildVerifier/2.0; +https://zevbuild.pages.dev)"
)

# Schemes that are ignored during disk file resolution
IGNORED_SCHEMES = ("javascript:", "mailto:", "tel:", "data:", "blob:", "sms:", "callto:")

# NSFW or prohibited domains to audit against in Tier 4
PROHIBITED_DOMAINS = ("hqporner.com", "pornhub.com", "xvideos.com")

# Color formatting helpers for terminal output
USE_COLOR = sys.stdout.isatty() and os.name != "nt" or "TERM" in os.environ

def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def red(t: str) -> str: return color(t, "31;1")
def green(t: str) -> str: return color(t, "32;1")
def yellow(t: str) -> str: return color(t, "33;1")
def blue(t: str) -> str: return color(t, "34;1")
def cyan(t: str) -> str: return color(t, "36;1")
def bold(t: str) -> str: return color(t, "1")


class HTMLAssetExtractor(HTMLParser):
    """
    Parses HTML documents to extract:
    1. DOM element IDs and anchor names (<tag id="...">, <a name="...">).
    2. Asset and navigation links (<a href>, <link href>, <script src>, <img src>, etc.).
    3. Document metadata (<title>, <link rel="icon">, <link rel="canonical">).
    """

    def __init__(self):
        super().__init__()
        self.dom_ids: Set[str] = set()
        self.links: List[Dict[str, any]] = []
        self.has_favicon: bool = False
        self.favicon_href: Optional[str] = None
        self.canonical_url: Optional[str] = None
        self.title: Optional[str] = None
        self._in_title: bool = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        attr_dict = {k.lower(): (v if v is not None else "") for k, v in attrs}
        line = self.getpos()[0]

        # Record DOM element IDs
        if "id" in attr_dict and attr_dict["id"].strip():
            self.dom_ids.add(attr_dict["id"].strip())

        # Support traditional <a name="..."> anchors
        if tag.lower() == "a" and "name" in attr_dict and attr_dict["name"].strip():
            self.dom_ids.add(attr_dict["name"].strip())

        # Metadata extraction
        if tag.lower() == "title":
            self._in_title = True

        if tag.lower() == "link":
            rel = attr_dict.get("rel", "").lower()
            href = attr_dict.get("href", "").strip()
            if "icon" in rel:
                self.has_favicon = True
                self.favicon_href = href
            if "canonical" in rel:
                self.canonical_url = href

        # Link and asset extraction across standard HTML tags
        # Tag -> Relevant attributes to verify
        tag_attrs_map = {
            "a": ["href"],
            "link": ["href"],
            "script": ["src"],
            "img": ["src"],
            "source": ["src"],
            "video": ["src", "poster"],
            "audio": ["src"],
            "iframe": ["src"],
            "embed": ["src"],
            "object": ["data"],
            "area": ["href"],
        }

        if tag.lower() in tag_attrs_map:
            for attr in tag_attrs_map[tag.lower()]:
                if attr in attr_dict:
                    val = attr_dict[attr].strip()
                    self.links.append({
                        "tag": tag.lower(),
                        "attr": attr,
                        "value": val,
                        "line": line,
                        "rel": attr_dict.get("rel", "").lower() if tag.lower() == "link" else None,
                    })

    def handle_endtag(self, tag: str):
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str):
        if self._in_title:
            self.title = (self.title or "") + data.strip()


class TierResult:
    """Encapsulates test outcomes for a single test tier."""
    def __init__(self, tier_num: int, name: str, description: str):
        self.tier_num = tier_num
        self.name = name
        self.description = description
        self.total_checks: int = 0
        self.passed_checks: int = 0
        self.failed_checks: int = 0
        self.warning_checks: int = 0
        self.failures: List[Dict[str, any]] = []
        self.warnings: List[Dict[str, any]] = []
        self.details: List[str] = []

    def record_pass(self, detail: str = ""):
        self.total_checks += 1
        self.passed_checks += 1
        if detail:
            self.details.append(f"[PASS] {detail}")

    def record_fail(self, source: str, target: str, reason: str, line: Optional[int] = None):
        self.total_checks += 1
        self.failed_checks += 1
        record = {
            "source": source,
            "target": target,
            "reason": reason,
            "line": line,
        }
        self.failures.append(record)
        loc = f"{source}:{line}" if line else source
        self.details.append(f"[FAIL] {loc} -> '{target}': {reason}")

    def record_warning(self, source: str, target: str, reason: str, line: Optional[int] = None):
        self.warning_checks += 1
        record = {
            "source": source,
            "target": target,
            "reason": reason,
            "line": line,
        }
        self.warnings.append(record)
        loc = f"{source}:{line}" if line else source
        self.details.append(f"[WARN] {loc} -> '{target}': {reason}")

    @property
    def passed(self) -> bool:
        return self.failed_checks == 0


class VerificationHarness:
    """
    Main verification harness coordinating document crawling,
    DOM analysis, 4-tier test execution, and reporting.
    """

    def __init__(self, project_root: Path, check_external: bool = True, verbose: bool = False):
        self.root = project_root.resolve()
        self.check_external = check_external
        self.verbose = verbose

        self.html_files: List[Path] = []
        self.doc_parsers: Dict[Path, HTMLAssetExtractor] = {}
        self.doc_contents: Dict[Path, str] = {}
        self.external_url_results: Dict[str, Dict[str, any]] = {}

    def crawl_documents(self):
        """Scans the repository for all HTML files excluding agent metadata and git dirs."""
        all_candidates = sorted(list(self.root.rglob("*.html")))
        self.html_files = [
            f for f in all_candidates
            if ".agents" not in f.parts
            and ".git" not in f.parts
            and "node_modules" not in f.parts
            and ".antigravity" not in f.parts
        ]

        for path in self.html_files:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                self.doc_contents[path] = content
                parser = HTMLAssetExtractor()
                if content:
                    parser.feed(content)
                self.doc_parsers[path] = parser
            except Exception as e:
                self.doc_contents[path] = ""
                self.doc_parsers[path] = HTMLAssetExtractor()

    def resolve_local_target(self, source_path: Path, raw_target: str) -> Tuple[Optional[Path], Optional[str], Optional[str]]:
        """
        Resolves a local href or src target relative to the source document or project root.
        Returns: (resolved_file_or_dir_path, fragment, error_message)
        """
        if not raw_target:
            return None, None, "Empty target value"

        # Split fragment and query string
        clean = raw_target.strip()
        frag_part = None
        if "#" in clean:
            clean, frag_part = clean.split("#", 1)

        if "?" in clean:
            clean = clean.split("?", 1)[0]

        clean = urllib.parse.unquote(clean.strip())

        # If target was only a fragment (e.g. #about), target file is source document itself
        if not clean:
            return source_path, frag_part, None

        # Root-relative path (e.g. /index.html or /favicon.svg)
        if clean.startswith("/"):
            resolved = (self.root / clean.lstrip("/")).resolve()
        else:
            resolved = (source_path.parent / clean).resolve()

        # If resolved points to a directory, check directory index
        if resolved.is_dir():
            index_candidate = resolved / "index.html"
            if index_candidate.is_file():
                return index_candidate, frag_part, None
            return resolved, frag_part, "Directory missing index.html"

        if resolved.is_file():
            return resolved, frag_part, None

        return resolved, frag_part, "File not found on disk"

    # =========================================================================
    # TIER 1: Feature Coverage (relative paths, asset loads, entry points)
    # =========================================================================
    def run_tier_1(self) -> TierResult:
        result = TierResult(
            1,
            "Feature Coverage",
            "Validates HTML discovery, non-empty files, local relative paths, static asset loads, and tool entry points"
        )

        # 1. HTML Discovery & Non-empty Check
        if not self.html_files:
            result.record_fail("Repository", "*.html", "No HTML files discovered in project root")
            return result

        result.record_pass(f"Discovered {len(self.html_files)} HTML documents across workspace")

        for doc_path in self.html_files:
            rel_name = doc_path.relative_to(self.root).as_posix()
            try:
                size = doc_path.stat().st_size
                if size == 0:
                    result.record_fail(rel_name, rel_name, "HTML document is 0 bytes (empty file)")
                else:
                    result.record_pass(f"{rel_name} exists and is non-empty ({size} bytes)")
            except Exception as e:
                result.record_fail(rel_name, rel_name, f"Failed to read file: {e}")

        # 2. Embedded Tools Entry Points
        required_tools = [
            ("Kalyan Matka Analytics", self.root / "tools" / "matka" / "index.html"),
            ("Spotify RetroWave Player", self.root / "tools" / "spotify" / "index.html"),
            ("Video Downloader Guide", self.root / "tools" / "download" / "index.html"),
            ("YouTube Downloader (v_yt)", self.root / "tools" / "yt" / "index.html"),
        ]

        for tool_name, tool_entry in required_tools:
            rel_tool = tool_entry.relative_to(self.root).as_posix()
            if not tool_entry.exists():
                result.record_fail("tools/", rel_tool, f"Required tool entry point '{tool_name}' missing")
            elif tool_entry.stat().st_size == 0:
                result.record_fail("tools/", rel_tool, f"Tool entry point '{tool_name}' is 0 bytes empty")
            else:
                result.record_pass(f"Tool entry point '{tool_name}' ({rel_tool}) verified")

        # 3. Universal Favicon & Core Metadata Audit
        for doc_path in self.html_files:
            rel_name = doc_path.relative_to(self.root).as_posix()
            parser = self.doc_parsers[doc_path]
            if not parser.has_favicon:
                result.record_warning(rel_name, "<head>", "Document lacks <link rel=\"icon\"> favicon tag")
            else:
                fav_target = parser.favicon_href or ""
                resolved_fav, _, fav_err = self.resolve_local_target(doc_path, fav_target)
                if fav_err:
                    result.record_fail(rel_name, fav_target, f"Favicon asset missing: {fav_err}")
                else:
                    result.record_pass(f"{rel_name} has valid favicon pointing to {resolved_fav.name}")

        # 4. Static Assets & Relative Paths Verification
        for doc_path in self.html_files:
            rel_source = doc_path.relative_to(self.root).as_posix()
            parser = self.doc_parsers[doc_path]

            for item in parser.links:
                val = item["value"]
                tag = item["tag"]
                attr = item["attr"]
                line = item["line"]

                # Empty attributes are handled in Tier 2
                if not val:
                    continue

                # Skip external protocols, mailto, tel, javascript, etc.
                if val.startswith(("http://", "https://")) or val.startswith(IGNORED_SCHEMES):
                    continue

                # Skip standalone fragments (checked in Tier 2)
                if val.startswith("#"):
                    continue

                resolved, _, err = self.resolve_local_target(doc_path, val)
                is_asset = tag in ("script", "img", "source", "video", "audio", "embed") or (
                    tag == "link" and item.get("rel") in ("stylesheet", "icon", "apple-touch-icon")
                )

                if err:
                    category = "Missing Static Asset" if is_asset else "Broken Relative Path"
                    result.record_fail(
                        rel_source,
                        val,
                        f"{category} (<{tag} {attr}=\"{val}\">) - {err}",
                        line
                    )
                else:
                    result.record_pass(f"{rel_source}:{line} -> <{tag} {attr}> resolves to {resolved.name}")

        return result

    # =========================================================================
    # TIER 2: Boundary & Corner Cases (empty values, anchors, query strings, ../ traversal)
    # =========================================================================
    def run_tier_2(self) -> TierResult:
        result = TierResult(
            2,
            "Boundary & Corner Cases",
            "Tests empty attributes, DOM fragment anchors, query string stripping, and multi-level parent ../ traversal"
        )

        for doc_path in self.html_files:
            rel_source = doc_path.relative_to(self.root).as_posix()
            parser = self.doc_parsers[doc_path]

            for item in parser.links:
                val = item["value"]
                tag = item["tag"]
                attr = item["attr"]
                line = item["line"]

                # 1. Empty or Whitespace Target Test
                if not val or not val.strip():
                    if tag == "img" and attr == "src":
                        result.record_warning(
                            rel_source,
                            f"<{tag} {attr}=\"\">",
                            "Dynamic preview image placeholder has empty src attribute",
                            line
                        )
                    else:
                        result.record_fail(
                            rel_source,
                            f"<{tag} {attr}=\"\">",
                            f"Empty or whitespace attribute value on <{tag} {attr}=\"\">",
                            line
                        )
                    continue

                # Ignore non-HTTP external or pseudo-schemes
                if val.startswith(IGNORED_SCHEMES):
                    result.record_pass(f"Ignored safe protocol scheme '{val[:15]}' in {rel_source}")
                    continue

                if val.startswith(("http://", "https://")):
                    continue

                # 2. In-Page DOM Fragment Anchors (#anchor)
                if val.startswith("#"):
                    frag = val[1:].strip()
                    # Lone '#' is a top-of-page return / standard skip link
                    if not frag:
                        result.record_pass(f"{rel_source}:{line} -> Valid top-of-page anchor '#'")
                        continue

                    if frag in parser.dom_ids:
                        result.record_pass(f"{rel_source}:{line} -> Anchor #{frag} matches element ID")
                    else:
                        result.record_fail(
                            rel_source,
                            val,
                            f"Broken DOM anchor #{frag} - no element with id='{frag}' found in {rel_source}",
                            line
                        )
                    continue

                # 3. Cross-Page Anchors & Path with Query Strings / Fragments
                resolved, frag, err = self.resolve_local_target(doc_path, val)
                if err:
                    # File level errors are captured in Tier 1; also note here if anchor involved
                    if frag:
                        result.record_fail(rel_source, val, f"Cannot verify anchor #{frag} because target {err}", line)
                    continue

                if frag:
                    # Target file exists; verify that target contains the anchor
                    target_parser = self.doc_parsers.get(resolved)
                    if not target_parser and resolved.is_file():
                        # Parse target if not already cached
                        try:
                            t_content = resolved.read_text(encoding="utf-8", errors="ignore")
                            target_parser = HTMLAssetExtractor()
                            target_parser.feed(t_content)
                            self.doc_parsers[resolved] = target_parser
                        except Exception:
                            target_parser = None

                    if target_parser and frag in target_parser.dom_ids:
                        result.record_pass(f"{rel_source}:{line} -> Cross-document anchor {resolved.name}#{frag} found")
                    else:
                        result.record_fail(
                            rel_source,
                            val,
                            f"Broken target anchor #{frag} in {resolved.relative_to(self.root).as_posix()}",
                            line
                        )

                # 4. Multi-Level Relative Traversal (../../) Safety Check
                if "../" in val:
                    # Check whether resolution tried to escape project root
                    try:
                        resolved.relative_to(self.root)
                        result.record_pass(f"{rel_source}:{line} -> Multi-level traversal '{val}' safely within workspace root")
                    except ValueError:
                        result.record_fail(
                            rel_source,
                            val,
                            f"Directory traversal escapes workspace root: {resolved}",
                            line
                        )

        return result

    # =========================================================================
    # TIER 3: Cross-Feature Combinations (hub-to-tool routing, tool-to-hub back links, 404 smart routing)
    # =========================================================================
    def run_tier_3(self) -> TierResult:
        result = TierResult(
            3,
            "Cross-Feature Combinations",
            "Enforces hub-to-tool navigation contracts, sub-tool back navigation breadcrumbs, and 404 smart keyword routing"
        )

        # 1. Main Hub (index.html) -> Tool Links Contract
        index_path = self.root / "index.html"
        if index_path.exists():
            index_content = self.doc_contents.get(index_path, "")
            # Verify index links to tools hub or embedded tools
            has_tools_link = (
                "tools/" in index_content or
                "tools/index.html" in index_content or
                "tools/matka/" in index_content
            )
            if has_tools_link:
                result.record_pass("Main Hub (index.html) contains active links to tools ecosystem")
            else:
                result.record_fail("index.html", "tools/", "Main Hub does not link to any tools directory")

        # 2. Tools Hub (tools/index.html) -> Sub-Tools Contract
        tools_index = self.root / "tools" / "index.html"
        if not tools_index.exists():
            result.record_fail("tools/", "tools/index.html", "Tools catalog hub missing")
        else:
            tools_content = self.doc_contents.get(tools_index, "")
            obsolete_patterns = [
                ("satta-matka-tools", "tools/matka/"),
                ("sp-ms-downloader", "tools/spotify/"),
                ("top-10-free-video-downloaders-in-india", "tools/download/"),
                ("v_yt", "tools/yt/"),
            ]
            for obs_name, correct_target in obsolete_patterns:
                # Detect obsolete links in href
                if f'href="{obs_name}' in tools_content or f"href='{obs_name}" in tools_content:
                    result.record_fail(
                        "tools/index.html",
                        obs_name,
                        f"Obsolete/dead tool directory referenced: '{obs_name}' (expected: '{correct_target}')"
                    )
                else:
                    result.record_pass(f"Tools hub does not contain obsolete directory link '{obs_name}'")

        # 3. Sub-Tool Back-Navigation Contract
        # Every page in tools/<tool_name>/ must have breadcrumb/back links to:
        # a) Main Hub: ../../index.html (or /index.html or /)
        # b) Tools Hub: ../index.html (or /tools/index.html or ../)
        sub_tool_dirs = ["matka", "spotify", "download", "yt"]
        for t_dir in sub_tool_dirs:
            tool_page = self.root / "tools" / t_dir / "index.html"
            rel_page = f"tools/{t_dir}/index.html"
            if not tool_page.exists() or tool_page.stat().st_size == 0:
                result.record_fail("tools/", rel_page, f"Sub-tool page '{rel_page}' is missing or empty")
                continue

            parser = self.doc_parsers.get(tool_page)
            all_hrefs = [item["value"] for item in parser.links if item["tag"] == "a" and item["attr"] == "href"]

            # Check back link to Main Hub
            has_home_link = any(
                h in ("../../index.html", "../../", "/index.html", "/", "https://zevbuild.pages.dev/", "https://zevbuild.pages.dev")
                for h in all_hrefs
            )
            # Check back link to Tools Hub
            has_tools_hub_link = any(
                h in ("../index.html", "../", "/tools/index.html", "/tools/", "https://zevbuild.pages.dev/tools/")
                for h in all_hrefs
            )

            if has_home_link:
                result.record_pass(f"{rel_page} has back-navigation link to Main Hub")
            else:
                result.record_fail(
                    rel_page,
                    "../../index.html",
                    "Missing back-navigation link to Main Hub (expected link to ../../index.html or /)"
                )

            if has_tools_hub_link:
                result.record_pass(f"{rel_page} has back-navigation link to Tools Hub")
            else:
                result.record_fail(
                    rel_page,
                    "../index.html",
                    "Missing back-navigation link to Tools Catalog (expected link to ../index.html or ../)"
                )

        # 4. Smart 404 Routing Contract
        page_404 = self.root / "404.html"
        if not page_404.exists():
            result.record_fail("Repository", "404.html", "Custom 404.html error handler page is missing")
        else:
            c_404 = self.doc_contents.get(page_404, "")
            # Check for obsolete route references in 404.html
            if "satta-matka-tools" in c_404:
                result.record_fail(
                    "404.html",
                    "tools/satta-matka-tools/index.html",
                    "404 page routes users to obsolete non-existent directory 'tools/satta-matka-tools/'"
                )
            else:
                result.record_pass("404.html does not contain references to obsolete 'satta-matka-tools'")

            # Check presence of home fallback button in 404.html
            if 'href="index.html"' in c_404 or 'href="/"' in c_404 or 'href="/index.html"' in c_404:
                result.record_pass("404.html contains home fallback navigation button")
            else:
                result.record_fail("404.html", "index.html", "404.html lacks a functional Home button")

        # 5. Sitemap.xml Alignment Check
        sitemap_path = self.root / "sitemap.xml"
        if sitemap_path.exists():
            sitemap_content = sitemap_path.read_text(encoding="utf-8", errors="ignore")
            # Extract URLs from <loc>
            urls = re.findall(r"<loc>(.*?)</loc>", sitemap_content)
            result.record_pass(f"sitemap.xml found with {len(urls)} entries")
            for u in urls:
                # Check for obsolete directories in sitemap
                for obs_name, _ in obsolete_patterns:
                    if obs_name in u:
                        result.record_fail("sitemap.xml", u, f"Sitemap references obsolete path '{obs_name}'")
        else:
            result.record_warning("Repository", "sitemap.xml", "sitemap.xml not found")

        return result

    # =========================================================================
    # TIER 4: Real-World Workload Scenarios (end-to-end user navigation walkthrough)
    # =========================================================================
    def run_tier_4(self) -> TierResult:
        result = TierResult(
            4,
            "Real-World Workload Scenarios",
            "Simulates real user end-to-end navigation flows, 404 recovery paths, content safety, and outbound HTTP reachability"
        )

        # Scenario A: New Visitor Navigation Walkthrough
        # Visitor lands on index.html, navigates primary menu anchors, clicks Launch Tools CTA
        index_doc = self.root / "index.html"
        if index_doc.exists():
            parser = self.doc_parsers[index_doc]
            primary_nav_anchors = ["products", "philosophy", "architecture", "specs", "contact"]
            for a in primary_nav_anchors:
                # Check if navigation link exists in index.html
                has_nav_link = any(item["value"] == f"#{a}" for item in parser.links)
                # Check if target ID exists in index.html
                has_target_id = a in parser.dom_ids

                if has_nav_link and not has_target_id:
                    result.record_fail(
                        "index.html",
                        f"#{a}",
                        f"Visitor navigation broken: Menu links to '#{a}' but section with id='{a}' does not exist"
                    )
                elif has_nav_link and has_target_id:
                    result.record_pass(f"Visitor navigation flow: '#{a}' anchor cleanly resolves to target section")

        # Scenario B: Tool Catalog Explorer Flow
        # Visitor explores tools/index.html and visits each sub-tool
        tools_doc = self.root / "tools" / "index.html"
        if tools_doc.exists():
            parser = self.doc_parsers[tools_doc]
            tool_links = [item["value"] for item in parser.links if item["tag"] == "a"]
            if not tool_links:
                result.record_fail("tools/index.html", "<a>", "Tools catalog has zero clickable navigation links")
            else:
                for tlink in tool_links:
                    if tlink.startswith(("http://", "https://")) or tlink.startswith(IGNORED_SCHEMES):
                        continue
                    resolved, _, err = self.resolve_local_target(tools_doc, tlink)
                    if err:
                        result.record_fail(
                            "tools/index.html",
                            tlink,
                            f"Catalog user gets 404 dead end when clicking tool link '{tlink}': {err}"
                        )
                    else:
                        result.record_pass(f"Catalog explorer click '{tlink}' successfully resolves to {resolved.name}")

        # Scenario C: 404 Recovery Journey
        # Simulates a user hitting a non-existent URL and recovering through 404 page
        page_404 = self.root / "404.html"
        if page_404.exists():
            c_404 = self.doc_contents.get(page_404, "")
            # Check keyword recovery targets in 404 JS/HTML
            # If 404 maps matka to satta-matka-tools, user falls into a 404 loop
            if "satta-matka-tools" in c_404:
                result.record_fail(
                    "404.html",
                    "tools/satta-matka-tools/index.html",
                    "User 404 recovery loop: smart redirect sends user to non-existent 'tools/satta-matka-tools/'"
                )
            else:
                result.record_pass("User 404 recovery does not trap user in dead redirect loop")

        # Scenario D: Prohibited Content & Adult Link Check
        for doc_path in self.html_files:
            rel_name = doc_path.relative_to(self.root).as_posix()
            parser = self.doc_parsers[doc_path]
            for item in parser.links:
                val = item["value"].lower()
                for bad_domain in PROHIBITED_DOMAINS:
                    if bad_domain in val:
                        result.record_fail(
                            rel_name,
                            item["value"],
                            f"Content Safety Policy Violation: Prohibited/NSFW domain '{bad_domain}' found in link",
                            item["line"]
                        )

        # Scenario E: Outbound External Link Reachability
        if self.check_external:
            self.execute_external_link_audit(result)
        else:
            result.record_pass("External HTTP/HTTPS link checks skipped (--local-only mode active)")

        return result

    def execute_external_link_audit(self, tier_result: TierResult):
        """Concurrently checks all unique outbound HTTP/HTTPS links across all documents."""
        unique_urls: Set[str] = set()
        url_locations: Dict[str, List[Tuple[str, int]]] = {}

        for doc_path in self.html_files:
            rel_name = doc_path.relative_to(self.root).as_posix()
            parser = self.doc_parsers[doc_path]
            for item in parser.links:
                val = item["value"]
                rel = (item.get("rel") or "").lower()
                if any(hint in rel for hint in ("preconnect", "dns-prefetch")):
                    continue
                if val.startswith(("http://", "https://")):
                    unique_urls.add(val)
                    url_locations.setdefault(val, []).append((rel_name, item["line"]))

        if not unique_urls:
            tier_result.record_pass("No outbound external HTTP/HTTPS links found")
            return

        print(cyan(f"[*] Auditing {len(unique_urls)} unique external outbound URLs (concurrency: {MAX_EXTERNAL_WORKERS})..."))

        def check_url(url: str) -> Tuple[str, int, Optional[str], str]:
            headers = {"User-Agent": BROWSER_USER_AGENT, "Accept": "*/*"}
            req = urllib.request.Request(url, headers=headers, method="HEAD")
            try:
                with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
                    return url, resp.status, None, resp.geturl()
            except urllib.error.HTTPError as e:
                # Some web servers (Cloudflare, GitHub, etc.) block HEAD requests with 403 or 405.
                # Fallback to GET with a small byte range.
                if e.code in (403, 405, 400):
                    try:
                        get_headers = {**headers, "Range": "bytes=0-1024"}
                        get_req = urllib.request.Request(url, headers=get_headers, method="GET")
                        with urllib.request.urlopen(get_req, timeout=DEFAULT_TIMEOUT) as get_resp:
                            return url, get_resp.status, None, get_resp.geturl()
                    except urllib.error.HTTPError as e2:
                        return url, e2.code, str(e2.reason), url
                    except Exception as e2:
                        return url, getattr(e2, "code", 999), str(e2), url
                return url, e.code, str(e.reason), url
            except urllib.error.URLError as e:
                return url, 998, str(e.reason), url
            except Exception as e:
                return url, 999, str(e), url

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_EXTERNAL_WORKERS) as executor:
            future_to_url = {executor.submit(check_url, u): u for u in unique_urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    target_url, status_code, err_msg, final_url = future.result()
                    self.external_url_results[url] = {
                        "status": status_code,
                        "error": err_msg,
                        "final_url": final_url,
                    }

                    locs = url_locations.get(url, [("Unknown", 0)])
                    first_loc = locs[0]

                    # 2xx and 3xx are clean passes
                    if 200 <= status_code < 400:
                        tier_result.record_pass(f"Outbound URL reachable (HTTP {status_code}): {url}")
                    elif status_code in (404, 410):
                        # Definite dead link
                        for doc_rel, line in locs:
                            tier_result.record_fail(
                                doc_rel,
                                url,
                                f"Dead external link returned HTTP {status_code} ({err_msg or 'Not Found'})",
                                line
                            )
                    elif status_code in (403, 429):
                        # Server anti-bot or rate limit - handle gracefully as warning
                        for doc_rel, line in locs:
                            tier_result.record_warning(
                                doc_rel,
                                url,
                                f"External server returned HTTP {status_code} (anti-bot / rate-limit protection)",
                                line
                            )
                    elif status_code == 998 and "getaddrinfo failed" in (err_msg or "").lower():
                        # DNS lookup failed
                        for doc_rel, line in locs:
                            tier_result.record_fail(
                                doc_rel,
                                url,
                                f"External domain unreachable (DNS lookup failure): {err_msg}",
                                line
                            )
                    else:
                        for doc_rel, line in locs:
                            tier_result.record_warning(
                                doc_rel,
                                url,
                                f"External URL returned status {status_code} ({err_msg})",
                                line
                            )
                except Exception as ex:
                    locs = url_locations.get(url, [("Unknown", 0)])
                    for doc_rel, line in locs:
                        tier_result.record_warning(doc_rel, url, f"External audit exception: {ex}", line)

    # =========================================================================
    # Suite Runner & Report Generators
    # =========================================================================
    def run_all(self, selected_tier: Optional[int] = None) -> List[TierResult]:
        """Runs the test suite across selected tiers (or all 4 tiers)."""
        self.crawl_documents()

        tier_runners = [
            (1, self.run_tier_1),
            (2, self.run_tier_2),
            (3, self.run_tier_3),
            (4, self.run_tier_4),
        ]

        results = []
        for t_num, runner in tier_runners:
            if selected_tier is None or selected_tier == t_num:
                res = runner()
                results.append(res)

        return results

    def print_summary(self, results: List[TierResult]):
        """Prints a human-readable, colorized terminal summary."""
        print("\n" + "=" * 78)
        print(bold(cyan("ZEVBUILD LINK & STATIC ASSET VERIFICATION SUITE — 4-TIER REPORT")))
        print("=" * 78)
        print(f"Project Root:        {self.root}")
        print(f"HTML Files Crawled:  {len(self.html_files)}")
        print(f"External Auditing:   {'Enabled' if self.check_external else 'Disabled (--local-only)'}")
        print("-" * 78)

        total_checks = sum(r.total_checks for r in results)
        total_passed = sum(r.passed_checks for r in results)
        total_failed = sum(r.failed_checks for r in results)
        total_warns = sum(r.warning_checks for r in results)

        for r in results:
            status_badge = green("[PASS]") if r.passed else red("[FAIL]")
            print(f"Tier {r.tier_num}: {r.name:<30} {status_badge}  "
                  f"(Pass: {r.passed_checks}/{r.total_checks}, Fail: {r.failed_checks}, Warn: {r.warning_checks})")

        print("=" * 78)
        overall_status = green("PASSED (100% HEALTHY)") if total_failed == 0 else red(f"FAILED ({total_failed} DEFECTS FOUND)")
        print(f"OVERALL STATUS: {overall_status}")
        print("=" * 78)

        # Print all failures grouped by tier
        all_failures = []
        for r in results:
            if r.failures:
                print(f"\n{bold(red(f'--- Tier {r.tier_num} Failures: {r.name} ---'))}")
                for f in r.failures:
                    loc = f"{f['source']}:{f['line']}" if f.get('line') else f['source']
                    print(f"  {red('✘')} {bold(loc)} -> '{f['target']}'")
                    print(f"     Reason: {f['reason']}")
                    all_failures.append(f)

        # Print warnings if verbose or present
        all_warnings = []
        for r in results:
            if r.warnings and (self.verbose or not r.passed):
                for w in r.warnings:
                    loc = f"{w['source']}:{w['line']}" if w.get('line') else w['source']
                    all_warnings.append(f"  {yellow('⚠')} {loc} -> '{w['target']}': {w['reason']}")

        if all_warnings and self.verbose:
            print(f"\n{bold(yellow('--- Warnings & Notices ---'))}")
            for line in all_warnings[:20]:
                print(line)
            if len(all_warnings) > 20:
                print(f"  ...and {len(all_warnings) - 20} more warnings.")

        print("\n" + "=" * 78 + "\n")

    def export_json(self, results: List[TierResult], output_path: Path):
        """Exports a machine-readable JSON failure/pass audit report."""
        report = {
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "project_root": str(self.root),
                "html_documents_count": len(self.html_files),
                "external_checked": self.check_external,
            },
            "summary": {
                "total_tiers": len(results),
                "passed_tiers": sum(1 for r in results if r.passed),
                "failed_tiers": sum(1 for r in results if not r.passed),
                "total_checks": sum(r.total_checks for r in results),
                "total_passed": sum(r.passed_checks for r in results),
                "total_failed": sum(r.failed_checks for r in results),
                "total_warnings": sum(r.warning_checks for r in results),
            },
            "tiers": [
                {
                    "tier_number": r.tier_num,
                    "name": r.name,
                    "description": r.description,
                    "passed": r.passed,
                    "total_checks": r.total_checks,
                    "passed_checks": r.passed_checks,
                    "failed_checks": r.failed_checks,
                    "warning_checks": r.warning_checks,
                    "failures": r.failures,
                    "warnings": r.warnings,
                }
                for r in results
            ],
            "external_url_results": self.external_url_results,
        }

        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(green(f"[+] Machine-readable report saved to {output_path}"))

    def export_markdown(self, results: List[TierResult], output_path: Path):
        """Exports an executive summary in Markdown format."""
        total_failed = sum(r.failed_checks for r in results)
        md_lines = [
            "# Zevbuild Verification Crawler & Link Audit Summary",
            f"\n- **Generated At**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            f"- **Target Root**: `{self.root}`",
            f"- **HTML Files Crawled**: {len(self.html_files)}",
            f"- **Mode**: {'External HTTP Included' if self.check_external else 'Local-Only (--local-only)'}",
            f"- **Overall Status**: **{'PASSED' if total_failed == 0 else 'FAILED (' + str(total_failed) + ' defects)'}**\n",
            "## Tier Breakdown\n",
            "| Tier | Name | Status | Checks | Passed | Failed | Warnings |",
            "|---|---|---|---|---|---|---|",
        ]

        for r in results:
            badge = "✅ PASS" if r.passed else "❌ FAIL"
            md_lines.append(
                f"| Tier {r.tier_num} | {r.name} | {badge} | {r.total_checks} | {r.passed_checks} | {r.failed_checks} | {r.warning_checks} |"
            )

        if total_failed > 0:
            md_lines.append("\n## Detected Defects\n")
            for r in results:
                if r.failures:
                    md_lines.append(f"### Tier {r.tier_num}: {r.name}\n")
                    for f in r.failures:
                        loc = f"{f['source']}:{f['line']}" if f.get("line") else f["source"]
                        md_lines.append(f"- **`{loc}`** -> `{f['target']}`: {f['reason']}")
                    md_lines.append("")

        output_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(green(f"[+] Markdown summary saved to {output_path}"))


def main():
    parser = argparse.ArgumentParser(
        description="Zevbuild Comprehensive Link, Asset, and Anchor Verification Suite"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Workspace root directory (defaults to directory containing script)",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "all"],
        help="Specific test tier to run (1, 2, 3, 4, or all)",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        default=False,
        help="Skip outbound external HTTP/HTTPS network checks",
    )
    parser.add_argument(
        "--check-external",
        action="store_true",
        default=False,
        help="Explicitly audit external outbound HTTP/HTTPS links (default behavior unless --local-only)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("audit_results.json"),
        help="Path for machine-readable JSON report output",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("audit_summary.md"),
        help="Path for Markdown executive summary output",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed verbose output for all checks and warnings",
    )

    args = parser.parse_args()

    # Determine whether external links should be checked:
    # If --local-only is passed, disable external checks.
    # If --check-external is passed, enable external checks.
    # If neither is passed, default to external checks enabled.
    check_external = True
    if args.local_only:
        check_external = False
    elif args.check_external:
        check_external = True

    selected_tier = int(args.tier) if args.tier != "all" else None

    harness = VerificationHarness(
        project_root=args.root,
        check_external=check_external,
        verbose=args.verbose,
    )

    results = harness.run_all(selected_tier=selected_tier)
    harness.print_summary(results)

    # Export reports
    json_path = args.json_output if args.json_output.is_absolute() else (args.root / args.json_output)
    md_path = args.markdown_output if args.markdown_output.is_absolute() else (args.root / args.markdown_output)

    harness.export_json(results, json_path)
    harness.export_markdown(results, md_path)

    # Exit code: 0 if all executed tiers passed with 0 defects, else 1
    has_failures = any(not r.passed for r in results)
    sys.exit(1 if has_failures else 0)


if __name__ == "__main__":
    main()
