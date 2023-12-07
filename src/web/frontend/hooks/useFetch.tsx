/* eslint-disable @typescript-eslint/no-explicit-any */
import axios from "axios";
import { RD } from "prelude";
import React from "react";

export const useFetch = <Data, Params extends Record<string, any> = object>({
  params,
  url,
}: {
  url: string;
  params?: Params;
}) => {
  const [data, setData] = React.useState<RD.RemoteData<string, Data>>(RD.Init);

  React.useEffect(() => {
    setData(RD.Loading);

    axios
      .get(url, { params })
      .then((x) => setData(RD.Data({ data: x.data })))
      .catch((x) => setData(RD.Error({ error: x })));
  }, [params, url]);

  return data;
};
