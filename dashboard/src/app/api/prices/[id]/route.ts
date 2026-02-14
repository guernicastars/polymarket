import { NextResponse } from "next/server";
import { getMarketPriceHistory } from "@/lib/queries";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const { searchParams } = new URL(request.url);

  const interval = searchParams.get("interval") === "1h" ? "1h" : "1m";
  const outcome = searchParams.get("outcome") ?? "Yes";

  try {
    const bars = await getMarketPriceHistory(id, outcome, interval);
    return NextResponse.json(bars);
  } catch (error) {
    console.error("Failed to fetch price history:", error);
    return NextResponse.json(
      { error: "Failed to fetch prices" },
      { status: 500 }
    );
  }
}
