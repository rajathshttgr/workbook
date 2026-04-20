## Category Examples

Use category filters to search dedicated indexes. Each category returns only that content type.

**Note:** Categories can be restrictive. If you're not getting enough results, try searching without a category first, then add one if needed.

### People Search (`category: "people"`)

Find people by role, expertise, or what they work on

```python
exa.search(
    "software engineer distributed systems",
    category="people",
    type="auto",
    num_results=10
)
```

**Tips:**

- Use SINGULAR form
- Describe what they work on
- No date/text filters supported

### Company Search (`category: "company"`)

Find companies by industry, criteria, or attributes

```python
exa.search(
    "AI startup healthcare",
    category="company",
    type="auto",
    num_results=10
)
```

**Tips:**

- Use SINGULAR form
- Simple entity queries
- Returns company objects, not articles

### News Search (`category: "news"`)

News articles

```python
exa.search_and_contents(
    "OpenAI announcements",
    category="news",
    type="auto",
    num_results=10,
    text={"max_characters": 20000}
)
```

**Tips:**

- Use livecrawl: "preferred" for breaking news
- Avoid date filters unless required

### Research Papers (`category: "research paper"`)

Academic papers

```python
exa.search_and_contents(
    "transformer architecture improvements",
    category="research paper",
    type="auto",
    num_results=10,
    text={"max_characters": 20000}
)
```

**Tips:**

- Use type: "auto" for most queries
- Includes arxiv.org, paperswithcode.com, and other academic sources

### Tweet Search (`category: "tweet"`)

Twitter/X posts

```python
exa.search_and_contents(
    "AI safety discussion",
    category="tweet",
    type="auto",
    num_results=10,
    text={"max_characters": 20000}
)
```

**Tips:**

- Good for real-time discussions
- Captures public sentiment

---

## Content Freshness (maxAgeHours)

`maxAgeHours` sets the maximum acceptable age (in hours) for cached content. If the cached version is older than this threshold, Exa will livecrawl the page to get fresh content.

| Value    | Behavior                                                    | Best For                                        |
| -------- | ----------------------------------------------------------- | ----------------------------------------------- |
| 24       | Use cache if less than 24 hours old, otherwise livecrawl    | Daily-fresh content                             |
| 1        | Use cache if less than 1 hour old, otherwise livecrawl      | Near real-time data                             |
| 0        | Always livecrawl (ignore cache entirely)                    | Real-time data where cached content is unusable |
| -1       | Never livecrawl (cache only)                                | Maximum speed, historical/static content        |
| _(omit)_ | Default behavior (livecrawl as fallback if no cache exists) | **Recommended** — balanced speed and freshness  |

**When LiveCrawl Isn't Necessary:**
Cached data is sufficient for many queries, especially for historical topics or educational content. These subjects rarely change, so reliable cached results can provide accurate information quickly.

See [maxAgeHours docs](https://exa.ai/docs/reference/livecrawling-contents#maxAgeHours) for more details.

---

## Other Endpoints

Beyond `/search`, Exa offers these endpoints:

| Endpoint    | Description                        | Docs                                               |
| ----------- | ---------------------------------- | -------------------------------------------------- |
| `/contents` | Get contents for known URLs        | [Docs](https://exa.ai/docs/reference/get-contents) |
| `/answer`   | Q&A with citations from web search | [Docs](https://exa.ai/docs/reference/answer)       |

**Example - Get contents for URLs:**

```json
POST /contents
{
  "urls": ["https://example.com/article"],
  "text": { "max_characters": 20000 }
}
```

---

## Troubleshooting

**Results not relevant?**

1. Try `type: "auto"` - most balanced option
2. Try `type: "deep"` - runs multiple query variations and ranks the combined results
3. Refine query - use singular form, be specific
4. Check category matches your use case

**Need structured data from search?**

1. Use `type: "deep"` or `type: "deep-reasoning"` with `outputSchema`
2. Define the fields you need in the schema — Exa returns grounded JSON with citations

**Results too slow?**

1. Use `type: "fast"`
2. Reduce `num_results`
3. Skip contents if you only need URLs

**No results?**

1. Remove filters (date, domain restrictions)
2. Simplify query
3. Try `type: "auto"` - has fallback mechanisms

---

## Resources

- Docs: https://exa.ai/docs
- Dashboard: https://dashboard.exa.ai
- API Status: https://status.exa.ai
