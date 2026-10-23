import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NASA-RUL — Predictive Maintenance Intelligence",
  description: "Mobile-first turbofan Remaining Useful Life (RUL) estimation using NASA C-MAPSS FD001 and PyTorch LSTM.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased bg-slate-50 text-slate-900 min-h-screen">
        {children}
      </body>
    </html>
  );
}
