export default {
  async fetch(request) {
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, HEAD, POST, OPTIONS",
      "Access-Control-Allow-Headers": "*",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: corsHeaders,
      });
    }

    const url = new URL(request.url);
    url.hostname = "ergo-pomodoro-posture-ai.onrender.com";
    url.protocol = "https:";

    const proxyHeaders = new Headers(request.headers);
    proxyHeaders.set("Host", url.hostname);

    const isHasBody = !["GET", "HEAD"].includes(request.method);
    const proxyRequest = new Request(url.toString(), {
      method: request.method,
      headers: proxyHeaders,
      body: isHasBody ? request.body : null,
      redirect: "follow",
    });

    try {
      const response = await fetch(proxyRequest);

      const newResponse = new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });

      for (const [key, value] of Object.entries(corsHeaders)) {
        newResponse.headers.set(key, value);
      }

      return newResponse;
    } catch (error) {
      return new Response(
        JSON.stringify({
          error: "Lỗi kết nối Render (Có thể Server đang khởi động / Cold Start)",
          details: error.message,
        }),
        {
          status: 502,
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders,
          },
        }
      );
    }
  },
};
