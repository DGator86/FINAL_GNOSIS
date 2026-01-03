# Super Gnosis SaaS Frontend

This is the control plane for the SaaS. It connects to the Python `web_api.py` backend.

## Recommended Stack
- **Framework**: Next.js 14 (App Router)
- **Auth**: Clerk (easiest for SaaS) or NextAuth.js
- **UI**: Tailwind CSS + Shadcn/ui
- **State**: TanStack Query (React Query)
- **Charts**: Recharts (for equity curves)

## Setup Instructions

1. **Initialize Project**
   ```bash
   npx create-next-app@latest . --typescript --tailwind --eslint
   npm install @tanstack/react-query axios lucide-react recharts
   ```

2. **Connect to Backend**
   Update `next.config.js` to proxy API requests to Python:
   ```js
   module.exports = {
     async rewrites() {
       return [
         {
           source: '/api/:path*',
           destination: 'http://localhost:8000/:path*',
         },
       ]
     },
   }
   ```

3. **Sample Dashboard Component (`app/dashboard/page.tsx`)**
   
   ```tsx
   "use client";
   
   import { useQuery } from "@tanstack/react-query";
   import axios from "axios";
   
   export default function Dashboard() {
     const { data: profile } = useQuery({
       queryKey: ["profile"],
       queryFn: async () => (await axios.get("/api/saas/profile")).data
     });
   
     const { data: summary } = useQuery({
       queryKey: ["summary"],
       queryFn: async () => (await axios.get("/api/saas/dashboard/summary")).data
     });
   
     if (!profile) return <div>Loading...</div>;
   
     return (
       <div className="p-8">
         <h1 className="text-2xl font-bold mb-4">Welcome, {profile.full_name}</h1>
         
         <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
           <div className="card p-4 border rounded shadow">
             <h3 className="text-gray-500">Tier</h3>
             <p className="text-xl font-bold uppercase">{profile.tier}</p>
           </div>
           
           <div className="card p-4 border rounded shadow">
             <h3 className="text-gray-500">Active Trades</h3>
             <p className="text-xl font-bold">{summary?.total_trades || 0}</p>
           </div>
           
           <div className="card p-4 border rounded shadow">
             <h3 className="text-gray-500">System Status</h3>
             <p className="text-xl font-bold text-green-500">{summary?.system_status}</p>
           </div>
         </div>
       </div>
     );
   }
   ```
