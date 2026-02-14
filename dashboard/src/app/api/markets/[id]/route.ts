import { NextResponse } from "next/server";
import { getMarketDetail, getMarketTrades } from "@/lib/queries";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  try {
    const [market, trades] = await Promise.all([
      getMarketDetail(id),
      getMarketTrades(id, 50),
    ]);

    if (!market) {
      return NextResponse.json({ error: "Market not found" }, { status: 404 });
    }

    return NextResponse.json({ market, trades });
  } catch (error) {
    console.error("Failed to fetch market detail:", error);
    return NextResponse.json(
      { error: "Failed to fetch market" },
      { status: 500 }
    );
  }
}
