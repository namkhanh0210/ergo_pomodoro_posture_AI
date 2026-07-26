export default {
  async fetch(request, env, ctx) {
    // 1. Cấu hình các Header CORS cấp phép cho Vercel (Front-end)
    const corsHeaders = {
      "Access-Control-Allow-Origin": "https://ergoandpostureai.vercel.app", // Có thể đổi thành "*" nếu muốn cho phép mọi nguồn
      "Access-Control-Allow-Methods": "GET, HEAD, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
    };

    // 2. Bắt buộc: Xử lý request thăm dò (OPTIONS preflight) từ trình duyệt
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204, // No Content
        headers: corsHeaders,
      });
    }

    // 3. Lấy URL gốc và đổi domain trỏ về Backend FastAPI thực sự của bạn
    const url = new URL(request.url);
    
    // ⚠️ QUAN TRỌNG: Thay domain dưới đây bằng domain Backend FastAPI (Hugging Face hoặc Render) của bạn
    url.hostname = "ergo-pomodoro-posture-ai.onrender.com"; 

    // Tạo request chuyển tiếp
    const proxyRequest = new Request(url.toString(), {
      method: request.method,
      headers: request.headers,
      body: request.body,
      redirect: "follow",
    });

    try {
      // 4. Gửi request đến Backend thật
      const response = await fetch(proxyRequest);

      // 5. Clone response và gắn thêm CORS Headers vào trước khi trả về cho Front-end
      const modifiedResponse = new Response(response.body, response);
      for (const [key, value] of Object.entries(corsHeaders)) {
        modifiedResponse.headers.set(key, value);
      }

      return modifiedResponse;
    } catch (error) {
      // Bắt lỗi nếu Backend thật bị sập hoặc timeout
      return new Response(JSON.stringify({ error: "Backend is unreachable", details: error.message }), {
        status: 502,
        headers: {
          "Content-Type": "application/json",
          ...corsHeaders
        }
      });
    }
  },
};