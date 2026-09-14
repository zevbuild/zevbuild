/**
 * Cloudflare Pages Function: /api/live-kalyan
 * Serverless Edge Worker running on Cloudflare Pages (https://zevbuild.pages.dev/api/live-kalyan)
 * Fetches live Kalyan results directly from dpbossx.net with zero CORS issues,
 * parses the live draw string, and returns real-time JSON with no-cache headers.
 */

export async function onRequest(context) {
  const corsHeaders = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
    "Content-Type": "application/json; charset=utf-8",
  };

  if (context.request.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  const targetUrl = "https://dpbossx.net/kalyan-chart.php";

  try {
    const response = await fetch(targetUrl, {
      headers: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
      },
      cf: {
        // Cloudflare fetch options: disable edge caching
        cacheTtl: 0,
        cacheEverything: false,
      },
    });

    if (!response.ok) {
      throw new Error(`Upstream returned HTTP ${response.status}`);
    }

    const html = await response.text();

    // 1. Extract live result from <div class="chart-result">...<span>RESULT</span>
    const resultMatch = html.match(/<div class="chart-result"[^>]*>[\s\S]*?<span>(.*?)<\/span>/i);
    const rawResult = resultMatch ? resultMatch[1].trim() : null;

    let openPanna = null;
    let openDigit = null;
    let closeDigit = null;
    let closePanna = null;
    let jodi = null;
    let status = "UNKNOWN";

    if (rawResult) {
      // Examples:
      // "560-15-249" -> Open Panna 560, Jodi 15 (Open 1, Close 5), Close Panna 249
      // "299-0" -> Open Panna 299, Open Digit 0, Close pending
      // "299-0*-***" -> Open declared, close pending
      const fullMatch = rawResult.match(/^(\d{3})-(\d{2})-(\d{3})$/);
      const partialMatch = rawResult.match(/^(\d{3})-(\d{1,2})(?:[\*-]*)?$/);

      if (fullMatch) {
        openPanna = fullMatch[1];
        jodi = fullMatch[2];
        openDigit = parseInt(jodi[0], 10);
        closeDigit = parseInt(jodi[1], 10);
        closePanna = fullMatch[3];
        status = "FULL_JODI_DECLARED";
      } else if (partialMatch) {
        openPanna = partialMatch[1];
        const digits = partialMatch[2];
        openDigit = parseInt(digits[0], 10);
        if (digits.length === 2) {
          closeDigit = parseInt(digits[1], 10);
          jodi = digits;
          status = "FULL_JODI_DECLARED";
        } else {
          closeDigit = null;
          jodi = digits + "*";
          status = "OPEN_DECLARED";
        }
      }
    }

    const payload = {
      status: "success",
      timestamp: new Date().toISOString(),
      raw_result: rawResult,
      draw: {
        open_panna: openPanna,
        open_digit: openDigit,
        close_digit: closeDigit,
        close_panna: closePanna,
        jodi: jodi,
        status: status,
      },
    };

    return new Response(JSON.stringify(payload), { headers: corsHeaders });
  } catch (err) {
    return new Response(
      JSON.stringify({
        status: "error",
        message: err.message,
        timestamp: new Date().toISOString(),
      }),
      { status: 502, headers: corsHeaders }
    );
  }
}
