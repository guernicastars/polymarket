import { NextResponse } from "next/server";
import { getTopMarkets } from "@/lib/queries";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = Math.min(
    parseInt(searchParams.get("limit") ?? "50", 10),
    200
  );
  const category = searchParams.get("category");

  try {
    let markets = await getTopMarkets(limit);

    if (category) {
      markets = markets.filter(
        (m) => m.category.toLowerCase() === category.toLowerCase()
      );
    }

    return NextResponse.json(markets);
  } catch (error) {
    console.error("Failed to fetch markets:", error);
    return NextResponse.json(
      { error: "Failed to fetch markets" },
      { status: 500 }
    );
  }
}
