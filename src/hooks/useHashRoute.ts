import { useEffect, useState } from "react";

function getRoute() {
  return window.location.hash.replace(/^#\/?/, "");
}

export function useHashRoute() {
  const [route, setRoute] = useState(getRoute);

  useEffect(() => {
    const handleHashChange = () => setRoute(getRoute());

    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  return route;
}
