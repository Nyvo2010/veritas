"""Source hygiene and citation contracts.

These are the behaviours the report depends on: one identity per page, one
number per source, and no citation marker that does not resolve to the
bibliography.

Run with: python -m unittest discover -s tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from models.research_state import SearchResult  # noqa: E402
from utils.sources import (  # noqa: E402
    build_source_registry,
    canonical_url,
    citation_count,
    clean_page_text,
    normalize_citations,
    number_for,
    prepare_results,
    quality_score,
    select_for_analysis,
    source_type,
)


def result(url, *, title="A useful page title", content="x" * 600,
           score=0.7, full_content=None, category="general", quality=0.0,
           published_date=None):
    return SearchResult(
        url=url,
        title=title,
        content=content,
        score=score,
        full_content=full_content,
        category=category,
        quality=quality,
        published_date=published_date,
    )


class CanonicalUrlTests(unittest.TestCase):
    def test_tracking_scheme_and_www_do_not_create_a_second_source(self):
        self.assertEqual(
            canonical_url("http://www.Example.com/Article/?utm_source=x&utm_medium=y#part"),
            canonical_url("https://example.com/Article"),
        )

    def test_path_case_is_preserved(self):
        self.assertNotEqual(
            canonical_url("https://example.com/Article"),
            canonical_url("https://example.com/article"),
        )

    def test_meaningful_query_parameters_are_preserved(self):
        self.assertNotEqual(
            canonical_url("https://example.com/page?page=2"),
            canonical_url("https://example.com/page"),
        )

    def test_invalid_url_is_not_a_source(self):
        self.assertEqual(canonical_url(""), "")
        self.assertEqual(canonical_url("not a url"), "")


class PrepareResultsTests(unittest.TestCase):
    def test_thin_and_blocked_pages_are_rejected(self):
        results = [
            result("https://example.com/thin", content="short", score=0.1, title="x"),
            result(
                "https://example.com/blocked",
                content="Access denied. Enable JavaScript to continue.",
                score=0.8,
            ),
            result("https://example.com/real", content="y" * 800),
        ]
        kept, rejected = prepare_results(results)
        self.assertEqual([r.url for r in kept], ["https://example.com/real"])
        self.assertEqual(len(rejected), 2)

    def test_a_result_below_the_relevance_floor_is_rejected(self):
        kept, rejected = prepare_results(
            [result("https://example.com/movie", score=0.05)]
        )
        self.assertEqual(kept, [])
        self.assertEqual(len(rejected), 1)

    def test_a_missing_score_is_not_read_as_irrelevant(self):
        kept, _ = prepare_results([result("https://example.com/page", score=0.0)])
        self.assertEqual(len(kept), 1)

    def test_a_long_blocked_page_is_rejected(self):
        blocked = result(
            "https://example.com/paywall",
            content="Access denied. " + "This article continues elsewhere. " * 200,
            score=0.9,
        )
        kept, rejected = prepare_results([blocked])
        self.assertEqual(kept, [])
        self.assertEqual(len(rejected), 1)

    def test_a_long_article_about_captchas_is_kept(self):
        article = "Why captchas exist\n\n" + "A real sentence about bot defence. " * 100
        kept, _ = prepare_results(
            [result("https://example.com/captchas", content=article)]
        )
        self.assertEqual([r.url for r in kept], ["https://example.com/captchas"])

    def test_a_block_phrase_quoted_mid_article_is_kept(self):
        article = "A study of access controls. " * 100 + " Access denied is the usual response."
        kept, _ = prepare_results(
            [result("https://example.com/access-controls", content=article)]
        )
        self.assertEqual(len(kept), 1)

    def test_login_and_policy_pages_are_not_sources(self):
        for url in ("https://example.com/login", "https://example.com/privacy-policy"):
            with self.subTest(url=url):
                kept, _ = prepare_results([result(url)])
                self.assertEqual(kept, [])

    def test_duplicate_url_keeps_the_richest_version(self):
        first = result("https://example.com/a", full_content="a" * 400)
        second = result("http://www.example.com/a/", full_content="b" * 1200, score=0.9)
        kept, rejected = prepare_results([first, second])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].full_content, "b" * 1200)
        self.assertEqual(kept[0].score, 0.9)
        self.assertEqual(len(rejected), 1)

    def test_syndicated_article_is_merged_by_content(self):
        text = "The same syndicated article body. " * 40
        first = result("https://one.example/story", full_content=text)
        second = result("https://two.example/story", full_content=text)
        kept, rejected = prepare_results([first, second])
        self.assertEqual(len(kept), 1)
        self.assertIn("duplicate article text", rejected[0][1])

    def test_a_republished_article_with_different_chrome_is_one_source(self):
        # The bodies are near-identical, but each site adds its own header,
        # byline and related-links block, so the first 320 characters differ.
        body = "Quantization cuts weight precision to four bits, which halves " \
               "memory traffic and changes the accuracy trade-off. " * 30
        first = result(
            "https://one.example/story",
            full_content="Home | One News | Subscribe now\n" + body,
        )
        second = result(
            "https://two.example/story",
            full_content="Two Daily - Sign in - Menu\n" + body + "\nRelated: other news",
        )

        kept, rejected = prepare_results([first, second])

        self.assertEqual(len(kept), 1)
        self.assertIn("near-duplicate article text", rejected[0][1])

    def test_the_same_paper_as_html_and_as_pdf_is_one_source(self):
        title = "A Comprehensive Evaluation of Quantization Strategies for Large Language Models"
        html = result(
            "https://arxiv.org/html/2402.16775v1", title=title,
            full_content="Quantization strategies survey body. " * 40,
        )
        pdf = result(
            "https://aclanthology.org/2024.findings-acl.726.pdf",
            title="[PDF] A Comprehensive Evaluation of Quantization Strategies for Large Language ...",
            full_content="Different rendering of the same survey. " * 40,
        )

        kept, rejected = prepare_results([html, pdf])

        self.assertEqual(len(kept), 1)
        self.assertIn("near-duplicate article text", rejected[0][1])

    def test_two_articles_on_the_same_topic_stay_separate(self):
        first = result(
            "https://one.example/a",
            full_content="Quantization reduces weight precision to save memory. " * 20,
        )
        second = result(
            "https://two.example/b",
            full_content="Speculative decoding drafts tokens with a small model. " * 20,
        )

        kept, _ = prepare_results([first, second])

        self.assertEqual(len(kept), 2)

    def test_quality_prefers_educational_over_social_all_else_equal(self):
        educational = result("https://example.com/paper", category="educational")
        social = result("https://example.com/post", category="social")
        self.assertGreater(quality_score(educational), quality_score(social))


class SourceTypeTests(unittest.TestCase):
    def test_page_kinds_are_distinguished(self):
        cases = {
            "https://www.youtube.com/watch?v=abc": "video",
            "https://www.facebook.com/page/videos/123": "video",
            "https://www.reddit.com/r/WaterTreatment/comments/1": "forum",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10054062": "academic",
            "https://www.mdpi.com/2305-6304/13/9/782": "academic",
            "https://www.cdc.gov/water/index.html": "institution",
            "https://news.osu.edu/some-bottled-water-worse-than-tap": "news",
            "https://med.stanford.edu/news/insights/2025/01/x.html": "institution",
            "https://en.wikipedia.org/wiki/Microplastic": "reference",
            "https://docs.python.org/3/library/re.html": "docs",
            "https://www.bbc.com/news/science": "news",
            "https://vendor.example/blog/post": "web",
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                self.assertEqual(source_type(url), expected)

    def test_a_video_scores_below_a_paper_at_equal_relevance(self):
        video = result(
            "https://www.youtube.com/watch?v=abc", score=0.9,
            title="A long and confident video title",
        )
        paper = result(
            "https://www.nature.com/articles/abc", score=0.9,
            title="A long and confident paper title",
        )
        self.assertGreater(quality_score(paper), quality_score(video))
        self.assertLess(quality_score(video), 0.7)

    def test_registry_entry_records_the_page_kind(self):
        _, entries = build_source_registry([
            result("https://www.youtube.com/watch?v=abc")
        ])
        self.assertEqual(entries[0].source_type, "video")


class CleanPageTextTests(unittest.TestCase):
    def test_images_and_link_urls_are_removed_but_labels_kept(self):
        raw = (
            '![logo](/img/s.png) [Read more](https://x.example/a) '
            '<div>Body text about filters</div>'
        )
        cleaned = clean_page_text(raw)
        self.assertNotIn("s.png", cleaned)
        self.assertNotIn("https://x.example/a", cleaned)
        self.assertIn("Read more", cleaned)
        self.assertIn("Body text about filters", cleaned)

    def test_navigation_boilerplate_is_dropped(self):
        raw = "Skip to content\nMenu\nReal evidence about filters.\nAccept all cookies"
        self.assertEqual(clean_page_text(raw), "Real evidence about filters.")

    def test_paragraph_structure_is_kept(self):
        raw = "First paragraph.\n\n\n\nSecond paragraph."
        self.assertEqual(clean_page_text(raw), "First paragraph.\n\nSecond paragraph.")

    def test_empty_input_is_empty_output(self):
        self.assertEqual(clean_page_text(None), "")


class AnalysisSelectionTests(unittest.TestCase):
    def test_stronger_evidence_is_ranked_first(self):
        weak = result("https://a.example/weak", score=0.9, quality=0.2)
        strong = result("https://b.example/strong", score=0.1, quality=0.95)

        picked = select_for_analysis([weak, strong], limit=2, per_domain=3)

        self.assertEqual([r.url for r in picked], [strong.url, weak.url])

    def test_one_domain_cannot_fill_the_evidence_list(self):
        results = [result(f"https://a.example/{i}", quality=0.9) for i in range(4)]
        results += [result(f"https://b.example/{i}", quality=0.5) for i in range(2)]
        results += [result(f"https://c.example/{i}", quality=0.4) for i in range(2)]

        picked = select_for_analysis(results, limit=5, per_domain=2)
        domains = [r.url.split("/")[2] for r in picked]

        self.assertEqual(len(picked), 5)
        self.assertEqual(domains.count("a.example"), 2)
        self.assertEqual(domains.count("b.example"), 2)
        self.assertEqual(domains.count("c.example"), 1)

    def test_overflow_fills_when_no_other_domain_exists(self):
        results = [
            result(f"https://only.example/{i}", quality=0.9 - i * 0.01)
            for i in range(4)
        ]

        picked = select_for_analysis(results, limit=3, per_domain=2)

        self.assertEqual(len(picked), 3)

    def test_equal_scores_keep_the_engine_order(self):
        first = result("https://a.example/1", quality=0.5, score=0.5)
        second = result("https://b.example/2", quality=0.5, score=0.5)

        picked = select_for_analysis([first, second], limit=2, per_domain=3)

        self.assertEqual([r.url for r in picked], [first.url, second.url])


class RegistryTests(unittest.TestCase):
    def test_numbers_are_stable_and_skip_unusable_sources(self):
        results = [
            result("https://bad.example/thin", content="short", score=0.1, title="x"),
            result("https://one.example/a", content="a" * 600),
            result("http://www.one.example/a/", content="b" * 600),
            result("https://two.example/b", content="c" * 600),
        ]
        numbers, entries = build_source_registry(results)
        self.assertEqual([entry.number for entry in entries], [1, 2])
        self.assertEqual([entry.url for entry in entries],
                         ["https://one.example/a", "https://two.example/b"])
        self.assertEqual(number_for("http://www.one.example/a/", numbers), 1)
        self.assertEqual(number_for("https://missing.example/", numbers), None)

    def test_registry_number_lookup_normalises_the_url(self):
        numbers, _ = build_source_registry([result("https://example.com/a?utm_source=x")])
        self.assertEqual(number_for("https://www.example.com/a", numbers), 1)


class CitationTests(unittest.TestCase):
    def test_adjacent_markers_lose_the_comma_between_them(self):
        self.assertEqual(
            normalize_citations("A claim [Source 1], [Source 2].", {1, 2}),
            "A claim [Source 1][Source 2].",
        )


    def test_markers_are_canonicalised_and_unknown_sources_dropped(self):
        text = (
            "First claim [source 1]. Second claim [Sources 2, 5]. "
            "Third claim [Source 1, Source 2]. Invented [Source 99]."
        )
        cleaned = normalize_citations(text, valid_numbers={1, 2, 3}, allowed_numbers={1, 2})
        self.assertIn("First claim [Source 1].", cleaned)
        self.assertIn("Second claim [Source 2].", cleaned)
        self.assertIn("Third claim [Source 1][Source 2].", cleaned)
        self.assertNotIn("Source 99", cleaned)

    def test_a_number_outside_the_section_evidence_is_removed(self):
        cleaned = normalize_citations(
            "A claim [Source 4].", valid_numbers={1, 2, 3, 4}, allowed_numbers={1, 2}
        )
        self.assertEqual(cleaned, "A claim.")

    def test_parenthesised_marker_is_converted(self):
        self.assertEqual(
            normalize_citations("A claim (Source 2).", valid_numbers={1, 2}),
            "A claim [Source 2].",
        )

    def test_summary_can_have_all_citations_removed(self):
        cleaned = normalize_citations(
            "A claim [Source 1][Source 2].", valid_numbers={1, 2}, allowed_numbers=set()
        )
        self.assertEqual(cleaned, "A claim.")

    def test_count_is_per_marker(self):
        self.assertEqual(citation_count("A [Source 1][Source 2]. B [Source 1]."), 3)


if __name__ == "__main__":
    unittest.main()
