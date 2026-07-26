export default {
  async fetch(request) {
    // 1. Cấu hình mở cửa cho TẤT CẢ các domain và header
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, HEAD, POST, OPTIONS",
      "Access-Control-Allow-Headers": "*",
    };

    // 2. Trả lời ngay lập tức nếu trình duyệt hỏi thăm dò (OPTIONS)
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders });
    }

    // 3. Trỏ về Backend Render
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
      
      // 4. Ép cứng header CORS vào kết quả trả về
      newResponse.headers.set("Access-Control-Allow-Origin", "*");
      newResponse.headers.set("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS");
      newResponse.headers.set("Access-Control-Allow-Headers", "*");
      
      return newResponse;
    } catch (e) {
      return new Response(JSON.stringify({ error: "Lỗi Proxy", details: e.message }), {
        status: 500,
        headers: { "Content-Type": "application/json", ...corsHeaders }
      });
    }
  }
};
