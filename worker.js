export default {
  async fetch(request) {
    // 1. Cấu hình Headers chấp nhận CORS tuyệt đối
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, HEAD, POST, OPTIONS",
      "Access-Control-Allow-Headers": "*",
    };

    // 2. Trả lời ngay lập tức request thăm dò (OPTIONS) từ trình duyệt
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: corsHeaders
      });
    }

    // 3. Chuyển tiếp Request sang Backend Render
    const url = new URL(request.url);
    url.hostname = "ergo-pomodoro-posture-ai.onrender.com";

    const proxyRequest = new Request(url.toString(), {
      method: request.method,
      headers: request.headers,
      body: request.body,
    });

    try {
      const response = await fetch(proxyRequest);
      const newResponse = new Response(response.body, response);

      // 4. Bổ sung CORS headers vào Response trả về cho Vercel
      for (const [key, value] of Object.entries(corsHeaders)) {
        newResponse.headers.set(key, value);
      }

      return newResponse;
    } catch (error) {
      return new Response(JSON.stringify({ error: "Lỗi kết nối Render", details: error.message }), {
        status: 502,
        headers: {
          "Content-Type": "application/json",
          ...corsHeaders
        }
      });
    }
  }
};
